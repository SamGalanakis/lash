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
use clap::Parser;
use dataset::{LongCoTQuestion, load_questions};
use lash_core::{
    BackgroundRuntimeHost, BuiltinToolResultProjectionPluginFactory, EmbeddedRuntimeHost,
    InputItem, LashRuntime, PersistedSessionState, PersistentRuntimeServices, PluginHost,
    RuntimeCoreConfig, RuntimePersistence, SessionEvent, SessionPolicy, StandardContextApproach,
    TokioSessionTaskExecutor, TurnInjectionBridge, TurnInputInjectionBridge,
};
use lash_cli::config::LashConfig;
use lash::{
    TurnInput,
    advanced::{EventSink, ExecutionMode, TurnContext, TurnFinish, TurnOutcome, TurnStop},
    plugins::{PluginFactory, PluginSession, PluginSpec, StaticPluginFactory},
    prompt::{
        PromptBuiltin, PromptSlot, PromptTemplate, PromptTemplateEntry, PromptTemplateSection,
    },
    provider::ProviderHandle,
    tools::{ToolCall, ToolDefinition, ToolExecutionMode, ToolProvider, ToolResult},
    usage::{SessionUsageReport, TokenLedgerEntry, TokenUsage, diff_usage_reports},
};
use lash_export::{ExportFormat, export};
use lash_llm_tools::LlmToolsPluginFactory;
use lash_mode_rlm::{
    BuiltinRlmModePluginFactory, RlmModePluginConfig, RlmPromptFeatures, RlmTurnInputExt,
};
use lash_provider_openai::OPENROUTER_BASE_URL;
use lash_sqlite_store::Store;
use lash_subagents::{
    CapabilityField, CapabilityOptionalField, CapabilityRegistry, CapabilitySpec,
    LocalSubagentHost, StaticCapability, SubagentHost, SubagentsPluginFactory,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

const STATE_ROOT: &str = ".benchmarks/longcot";

/// The only text sent as the user message. The full problem is bound as
/// `input.prompt` in RLM globals — the model can interrogate it however it
/// wants (peek at a slice, measure length, hand a chunk to `spawn_agent`,
/// `observe` it wholesale, etc.). This directive intentionally does not
/// force a single strategy; that's lash RLM's raison d'être.
const LONGCOT_USER_DIRECTIVE: &str = concat!(
    "Solve the LongCoT problem bound as `input.prompt`. ",
    "Its length is reported in the Bound Variables section — decide how much of it to pull into context at once and use lashlang (slicing, `spawn_agent`, etc.) to decompose if helpful. ",
    "Follow every instruction in the problem exactly, including any ban on tools or code beyond what you need to inspect the input. ",
    "End by calling `submit <string>` from a fenced `lashlang` block, where the submitted string's final line is exactly `solution = <value>` matching the shape the problem specifies."
);

// Defaults. GPT-5.2 (`openai/gpt-5.2` via OpenRouter's OpenAI-
// compatible endpoint) is the working target. Model knobs mirror the
// upstream `src/configs/oai_gpt52.yaml`: `reasoning.effort=high` and
// `max_output_tokens=125000`. Matching those keeps our numbers
// directly comparable to the published leaderboard. `max_turns=50`
// still lines up with the reference RLM iteration cap; the
// execution engine is always lash's `lashlang`-backed RLM.
const DEFAULT_MODEL: &str = "openai/gpt-5.2";
const DEFAULT_VARIANT: &str = "high";
const DEFAULT_MAX_TURNS: usize = 50;
const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 125_000;
const DEFAULT_MAX_CONTEXT_TOKENS: usize = 1_000_000;
const DEFAULT_BATCH_SIZE: usize = 4;
const DEFAULT_HARNESS: &str = "restricted";

/// LongCoT runs are always in lashlang RLM mode. Standard mode is intentionally
/// not wired in here; the benchmark forbids external tools, so the broader
/// standard-mode plugin set (rolling history, observational memory, monitor,
/// task controls) would only inflate the prompt without adding capability.
const EXECUTION_MODE_LABEL: &str = "rlm";

/// Capability the model can pass to `spawn_agent`. Children inherit the root
/// model/variant and the same locked-down longcot tool surface.
const SUBAGENT_CAPABILITY: &str = "default";

#[derive(Parser, Debug, Clone)]
#[command(name = "bench-longcot")]
#[command(about = "Run LongCoT (Motwani et al., 2026) through Lash.")]
struct Args {
    // Selection — matches upstream run_inference.py flag names.
    #[arg(long, value_parser = ["easy", "medium", "hard", "longcot-mini", "longcot"])]
    difficulty: Option<String>,

    #[arg(long, value_parser = ["logic", "cs", "chemistry", "chess", "math"])]
    domain: Vec<String>,

    #[arg(long)]
    question_id: Vec<String>,

    #[arg(long)]
    max_questions: Option<usize>,

    #[arg(long, default_value_t = 0)]
    offset: usize,

    #[arg(long)]
    shuffle_seed: Option<u64>,

    // Run identity / output.
    #[arg(long)]
    run_id: Option<String>,

    #[arg(long, default_value = "lash")]
    run_name: String,

    #[arg(long)]
    output_dir: Option<PathBuf>,

    #[arg(long)]
    resume: bool,

    // Provider / model wiring.
    #[arg(long, default_value = DEFAULT_MODEL)]
    model: String,

    #[arg(long, default_value = "openai-compatible")]
    provider_id: String,

    /// Reasoning effort variant. Upstream's `oai_gpt52.yaml` pins this
    /// to `high`; we default to the same so our numbers sit on the
    /// same axis as the published leaderboard.
    #[arg(long, default_value = DEFAULT_VARIANT)]
    variant: String,

    #[arg(long)]
    api_key: Option<String>,

    #[arg(long)]
    base_url: Option<String>,

    /// Which longcot.ai leaderboard column this run is submitted to.
    /// Both values produce the same execution (lash RLM + lashlang +
    /// subagents — no per-question solver code either way); the flag
    /// only changes what goes in the manifest so the submission PR
    /// lands in the intended column. "Raw LLM" is deliberately not
    /// exposed — lash shouldn't pretend to be a bare-API-call entrant.
    #[arg(long, default_value = DEFAULT_HARNESS, value_parser = ["restricted", "open"])]
    harness: String,

    // Session policy.
    #[arg(long, default_value_t = DEFAULT_MAX_TURNS)]
    max_turns: usize,

    #[arg(long, default_value_t = DEFAULT_MAX_CONTEXT_TOKENS)]
    max_context_tokens: usize,

    /// Per-response max output tokens. Sets `LASH_MAX_OUTPUT_TOKENS` for
    /// this process (overrides the lash default of 32768). Matches
    /// upstream's `oai_gpt52.yaml` cap of 125_000.
    #[arg(long, default_value_t = DEFAULT_MAX_OUTPUT_TOKENS)]
    max_output_tokens: u64,

    // Execution.
    #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
    batch_size: usize,

    #[arg(long)]
    await_background_work: bool,

    #[arg(long)]
    dry_run: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RunManifest {
    run_id: String,
    created_at: String,
    model: String,
    provider_id: String,
    variant: Option<String>,
    base_url: String,
    /// Leaderboard column this run targets on longcot.ai: `restricted`
    /// or `open`. Both run the same lash RLM config; only the manifest
    /// label differs.
    harness: String,
    execution_mode: String,
    max_turns: usize,
    max_context_tokens: usize,
    max_output_tokens: u64,
    selection: SelectionSnapshot,
    selected_count: usize,
    responses_path: String,
    reference: ReferenceSettings,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SelectionSnapshot {
    domains: Vec<String>,
    difficulty: Option<String>,
    question_ids: Vec<String>,
    offset: usize,
    max_questions: Option<usize>,
    shuffle_seed: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ReferenceSettings {
    upstream_repo: String,
    reference_blog: String,
    reference_framework: String,
    note: String,
}

impl Default for ReferenceSettings {
    fn default() -> Self {
        Self {
            upstream_repo: "https://github.com/LongHorizonReasoning/longcot".to_string(),
            reference_blog: "https://raw.works/longcot-a-benchmark-worthy-of-a-rlms-attention/"
                .to_string(),
            reference_framework: "reference RLM runtime".to_string(),
            note:
                "This harness runs LongCoT through lash's lashlang-backed RLM. The iteration cap \
                 (50) and max-output cap (125k) match upstream's oai_gpt52.yaml reference config; \
                 the execution engine is lash, not the upstream raw-LLM harness."
                    .to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct QuestionResult {
    question_id: String,
    domain: String,
    difficulty: String,
    successful: bool,
    response_text: String,
    model: String,
    usage: SessionUsageReport,
    attempts: usize,
    elapsed_seconds: f64,
    iterations: usize,
    solution_line_present: bool,
    status: String,
    done_reason: String,
    failure_reason: Option<String>,
    lash: LashRunSnapshot,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LashRunSnapshot {
    execution_mode: String,
    variant: Option<String>,
    max_turns: usize,
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
    successful: usize,
    failed: usize,
    solution_line_present: usize,
    by_domain: BTreeMap<String, DomainBucket>,
    iterations: usize,
    usage: SessionUsageReport,
    responses_path: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct DomainBucket {
    count: usize,
    successful: usize,
    solution_line_present: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();
    // Register every built-in provider factory (codex, anthropic, openai-…)
    // before anyone calls `LashConfig::build_active_provider`. Without this,
    // providers loaded from `~/.lash/config.json` fail with "provider X is
    // not registered."
    lash_providers_builtin::register_all();

    let args = Args::parse();

    // LASH_MAX_OUTPUT_TOKENS is read inside the openrouter adapter for each
    // request. Setting it here (before any provider call) controls the cap for
    // every child task in this process.
    //
    // SAFETY: set_var must be called on a single thread before tokio spawns
    // workers. `main` is the first code path and has no concurrent readers.
    #[allow(unsafe_code)]
    unsafe {
        env::set_var("LASH_MAX_OUTPUT_TOKENS", args.max_output_tokens.to_string());
    }

    let state_root = PathBuf::from(STATE_ROOT);
    let vendor_dir = state_root.join("vendor").join("longcot");
    let data_dir = vendor_dir.join("src").join("data");
    if !data_dir.is_dir() {
        bail!(
            "LongCoT dataset not found under {} — run bench/longcot/setup.sh first",
            data_dir.display()
        );
    }
    let runs_dir = state_root.join("runs");
    fs::create_dir_all(&runs_dir).with_context(|| format!("create {}", runs_dir.display()))?;

    let questions = select_questions(
        load_questions(&data_dir, &args.domain, args.difficulty.as_deref())?,
        &args,
    );
    if questions.is_empty() {
        bail!("no LongCoT questions selected");
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
    let responses_dir = output_dir.join("responses");
    fs::create_dir_all(&responses_dir)
        .with_context(|| format!("create {}", responses_dir.display()))?;

    let execution_mode = ExecutionMode::new(EXECUTION_MODE_LABEL);
    let model_slug = args.model.replace(['/', ':'], "_");
    let domain_label = summarize_domain_selection(&args.domain);
    let diff_label = args.difficulty.clone().unwrap_or_else(|| "all".to_string());
    let responses_path = responses_dir.join(format!(
        "{domain_label}_{diff_label}_{run}-{model_slug}.jsonl",
        run = args.run_name,
    ));

    let manifest = RunManifest {
        run_id: run_id.clone(),
        created_at: Utc::now().to_rfc3339(),
        model: args.model.clone(),
        provider_id: args.provider_id.clone(),
        variant: Some(args.variant.clone()),
        base_url: resolve_base_url(&args),
        harness: args.harness.clone(),
        execution_mode: EXECUTION_MODE_LABEL.to_string(),
        max_turns: args.max_turns,
        max_context_tokens: args.max_context_tokens,
        max_output_tokens: args.max_output_tokens,
        selection: SelectionSnapshot {
            domains: if args.domain.is_empty() {
                dataset::DOMAINS.iter().map(|d| (*d).to_string()).collect()
            } else {
                args.domain.clone()
            },
            difficulty: args.difficulty.clone(),
            question_ids: args.question_id.clone(),
            offset: args.offset,
            max_questions: args.max_questions,
            shuffle_seed: args.shuffle_seed,
        },
        selected_count: questions.len(),
        responses_path: responses_path.display().to_string(),
        reference: ReferenceSettings::default(),
    };
    write_json(&output_dir.join("manifest.json"), &manifest)?;

    if args.dry_run {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "manifest": manifest,
                "questions": questions
                    .iter()
                    .map(|q| json!({
                        "question_id": q.question_id,
                        "domain": q.domain,
                        "difficulty": q.difficulty,
                        "prompt_len": q.prompt.chars().count(),
                    }))
                    .collect::<Vec<_>>(),
            }))?
        );
        return Ok(());
    }

    let provider = resolve_provider(&args)?;

    let completed = if args.resume {
        load_completed_ids(&responses_path)?
    } else {
        BTreeSet::new()
    };
    let pending = questions
        .iter()
        .filter(|q| !completed.contains(&q.question_id))
        .cloned()
        .collect::<Vec<_>>();

    eprintln!("LongCoT run_id={run_id}");
    eprintln!("  selected:         {}", questions.len());
    eprintln!("  pending:          {}", pending.len());
    eprintln!("  model:            {}", args.model);
    eprintln!("  execution-mode:   {}", manifest.execution_mode);
    eprintln!("  max_turns:        {}", args.max_turns);
    eprintln!("  max_output_tokens:{}", args.max_output_tokens);
    eprintln!("  batch_size:       {}", args.batch_size.max(1));
    eprintln!("  responses:        {}", responses_path.display());
    if !completed.is_empty() {
        eprintln!("  resuming:         skipping {} ids", completed.len());
    }

    if pending.is_empty() {
        eprintln!("nothing to run — responses file already covers every selected question");
        return Ok(());
    }

    let started_at = Utc::now();
    let started_instant = std::time::Instant::now();
    let semaphore = Arc::new(Semaphore::new(args.batch_size.max(1)));
    let provider = Arc::new(provider);
    let args_shared = Arc::new(args.clone());
    let output_dir_shared = Arc::new(output_dir.clone());
    let responses_path_shared = Arc::new(responses_path.clone());
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
        let responses_path = responses_path_shared.clone();
        let execution_mode = execution_mode.clone();
        join_set.spawn(async move {
            let _permit = permit;
            let result = run_question(
                output_dir.as_ref(),
                provider.as_ref(),
                args.as_ref(),
                execution_mode,
                question,
            )
            .await;
            if let Ok(row) = &result {
                let _ = append_response_row(responses_path.as_ref(), row);
            }
            (index, result)
        });
    }

    let mut indexed_results: Vec<(usize, QuestionResult)> = Vec::new();
    let mut completed_count = 0usize;
    while let Some(joined) = join_set.join_next().await {
        let (index, result) = match joined {
            Ok(value) => value,
            Err(err) => {
                join_set.abort_all();
                return Err(anyhow::anyhow!("benchmark task panicked: {err}"));
            }
        };
        let result = match result {
            Ok(value) => value,
            Err(err) => {
                join_set.abort_all();
                return Err(err);
            }
        };
        completed_count += 1;
        eprintln!(
            "  [{}/{}] {} [{}/{}] status={} solution_line={} t={:.1}s iters={}",
            completed_count,
            total,
            result.question_id,
            result.domain,
            result.difficulty,
            if result.successful {
                "ok"
            } else if matches!(result.failure_reason.as_deref(), Some("timed_out")) {
                "TIMEOUT"
            } else {
                "FAIL"
            },
            if result.solution_line_present {
                "y"
            } else {
                "n"
            },
            result.elapsed_seconds,
            result.iterations,
        );
        indexed_results.push((index, result));
    }
    indexed_results.sort_by_key(|(idx, _)| *idx);
    let results = indexed_results
        .into_iter()
        .map(|(_, r)| r)
        .collect::<Vec<_>>();

    let finished_at = Utc::now();
    let summary = RunSummary {
        run_id: run_id.clone(),
        started_at: started_at.to_rfc3339(),
        finished_at: finished_at.to_rfc3339(),
        duration_seconds: (finished_at - started_at).num_seconds(),
        question_count: questions.len(),
        result_count: results.len(),
        successful: results.iter().filter(|r| r.successful).count(),
        failed: results.iter().filter(|r| !r.successful).count(),
        solution_line_present: results.iter().filter(|r| r.solution_line_present).count(),
        by_domain: aggregate_by_domain(&results),
        iterations: results.iter().map(|r| r.iterations).sum(),
        usage: aggregate_usage(results.iter().map(|r| r.usage.clone())),
        responses_path: responses_path.display().to_string(),
    };
    write_json(&output_dir.join("results.json"), &summary)?;
    write_trace_index(&output_dir, &run_id, &results)?;

    let elapsed = started_instant.elapsed().as_secs_f64();
    eprintln!();
    eprintln!("Run summary:");
    eprintln!("  run_dir:              {}", output_dir.display());
    eprintln!("  responses:            {}", responses_path.display());
    eprintln!(
        "  successful:           {}/{}",
        summary.successful, summary.result_count
    );
    eprintln!(
        "  solution_line_present:{}/{}",
        summary.solution_line_present, summary.result_count
    );
    eprintln!("  iterations_total:     {}", summary.iterations);
    eprintln!("  wall_clock:           {elapsed:.1}s");
    eprintln!();
    eprintln!("Evaluate with:");
    eprintln!("  bench/longcot/evaluate.sh {}", output_dir.display());
    Ok(())
}

fn select_questions(mut questions: Vec<LongCoTQuestion>, args: &Args) -> Vec<LongCoTQuestion> {
    if !args.question_id.is_empty() {
        let wanted: BTreeSet<&str> = args.question_id.iter().map(String::as_str).collect();
        questions.retain(|q| wanted.contains(q.question_id.as_str()));
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
    // Tiny deterministic PRNG (splitmix64) for reproducible shuffles without
    // pulling in the `rand` crate.
    let mut state = seed;
    let n = items.len();
    for i in (1..n).rev() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let j = (z as usize) % (i + 1);
        items.swap(i, j);
    }
}

fn summarize_domain_selection(domains: &[String]) -> String {
    if domains.is_empty() {
        return "all".to_string();
    }
    let mut sorted: Vec<String> = domains.to_vec();
    sorted.sort();
    sorted.join("+")
}

async fn run_question(
    output_dir: &Path,
    provider: &ProviderHandle,
    args: &Args,
    execution_mode: ExecutionMode,
    question: LongCoTQuestion,
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

    let policy = SessionPolicy {
        model: args.model.clone(),
        provider: provider.clone(),
        max_context_tokens: Some(args.max_context_tokens),
        execution_mode: execution_mode.clone(),
        standard_context_approach: None,
        model_variant: Some(args.variant.clone()),
        max_turns: Some(args.max_turns),
        ..SessionPolicy::default()
    };

    let plugin_session = build_plugin_session(execution_mode.clone(), &policy)?;
    let services = PersistentRuntimeServices::new_with_bridges(
        plugin_session,
        TurnInjectionBridge::new(),
        TurnInputInjectionBridge::new(),
        store.clone() as Arc<dyn RuntimePersistence>,
    );
    let host = BackgroundRuntimeHost::new(
        EmbeddedRuntimeHost::new(
            RuntimeCoreConfig::default()
                .with_trace_jsonl_path(Some(trace_path.clone()))
                .with_prompt_template(longcot_prompt_template()),
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

    let before_usage = runtime.usage_report();
    let started = std::time::Instant::now();
    let cancel = tokio_util::sync::CancellationToken::new();
    let sink = Arc::new(LongCoTEventSink::new(question_dir.join("events.jsonl"))?);
    let sink_trait: Arc<dyn EventSink> = sink.clone();
    let turn = runtime
        .stream_turn(
            (TurnInput {
                items: vec![InputItem::Text {
                    text: LONGCOT_USER_DIRECTIVE.to_string(),
                }],
                image_blobs: Default::default(),
                mode: None,
                mode_turn_options: None,
                trace_turn_id: None,
                mode_extension: None,
                turn_context: TurnContext::default(),
            })
            .rlm_project(build_projected_bindings(&question)?)?,
            sink_trait.as_ref(),
            cancel,
        )
        .await
        .context("run longcot question")?;
    if args.await_background_work {
        runtime.await_background_work().await?;
    }
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let after_usage = runtime.usage_report();
    let usage = diff_usage_reports(&before_usage, &after_usage)
        .map(|rows| SessionUsageReport::from_entries(&rows))
        .map_err(anyhow::Error::msg)
        .context("diff usage reports")?;

    let partial_output = sink
        .last_llm_response()
        .or_else(|| non_empty(&turn.assistant_output.safe_text));
    let status = turn_status_label(&turn.outcome).to_string();
    let done_reason = done_reason_label(&turn.outcome).to_string();
    let successful = turn_completed(&turn.outcome);
    let response_text = partial_output.clone().unwrap_or_default();
    let failure_reason = if successful {
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
        format!("{response_text}\n"),
    )
    .with_context(|| format!("write {}", question_dir.join("answer.txt").display()))?;
    let solution_line_present = has_solution_line(&response_text);

    let result = QuestionResult {
        question_id: question.question_id.clone(),
        domain: question.domain.clone(),
        difficulty: question.difficulty.clone(),
        successful,
        response_text,
        model: args.model.clone(),
        usage,
        attempts: 1,
        elapsed_seconds,
        iterations: sink.iteration_count(),
        solution_line_present,
        status,
        done_reason,
        failure_reason,
        lash: LashRunSnapshot {
            execution_mode: EXECUTION_MODE_LABEL.to_string(),
            variant: Some(args.variant.clone()),
            max_turns: args.max_turns,
            max_output_tokens: args.max_output_tokens,
        },
    };
    write_json(&question_dir.join("result.json"), &result)?;

    // Emit a self-contained HTML trace alongside the session db. Failures
    // here should not take down the benchmark — traces are a debugging aid.
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

    // Project the actual outgoing system prompt out of the typed trace so you can
    // see exactly what the model was told in a small sidecar file.
    if let Err(err) = write_system_prompt_snapshot(&trace_path, &question_dir) {
        eprintln!(
            "warn: failed to snapshot system prompt for {}: {err:#}",
            question.question_id
        );
    }

    Ok(result)
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
        let Some(messages) = request_value.get("messages").and_then(|v| v.as_array()) else {
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

/// Minimal RLM plugin stack matching the continual-learning-bench pattern.
///
/// Registered tools (model-visible): `llm_query`, `continue_as`,
/// `list_async_handles`, `spawn_agent` (capability `default`).
///
/// Deliberately not registered:
/// - `monitor`, `tasks_list`, `tasks_stop` — there is no shell or background
///   work to watch in a pure-reasoning bench.
/// - Standard mode + rolling-history / observational-memory plugins — RLM is
///   the only execution path here; standard-mode contributions would only
///   inflate the prompt without adding capability.
/// - Filesystem, shell, search, web, editing, and MCP tools — LongCoT
///   explicitly forbids external tool use.
///
/// Children spawned via `spawn_agent` inherit the same explicit tool surface
///, so recursive descents stay inside
/// the locked-down set instead of accidentally picking up whatever happens to
/// be registered at the root.
fn build_plugin_session(
    execution_mode: ExecutionMode,
    policy: &SessionPolicy,
) -> anyhow::Result<Arc<PluginSession>> {
    let _longcot_tools = longcot_tool_definitions();
    let factories: Vec<Arc<dyn PluginFactory>> = vec![
        Arc::new(BuiltinToolResultProjectionPluginFactory::default()),
        Arc::new(BuiltinRlmModePluginFactory::new(RlmModePluginConfig {
            prompt_features: RlmPromptFeatures {
                images: false,
                ..RlmPromptFeatures::default()
            },
            ..RlmModePluginConfig::default()
        })),
        Arc::new(LlmToolsPluginFactory),
        Arc::new(StaticPluginFactory::new(
            "longcot_async_handles",
            PluginSpec::new().with_tool_provider(Arc::new(LongCoTAsyncHandlesTool)),
        )),
        Arc::new(SubagentsPluginFactory::new(
            policy.clone(),
            Arc::new(
                CapabilityRegistry::new().with(Arc::new(StaticCapability::new(
                    SUBAGENT_CAPABILITY,
                    CapabilitySpec {
                        model: CapabilityField::Inherit,
                        model_variant: CapabilityOptionalField::Inherit,
                        execution_mode: CapabilityField::Inherit,
                    },
                ))),
            ),
            Arc::new(LocalSubagentHost::default()) as Arc<dyn SubagentHost>,
        )),
    ];
    PluginHost::new(factories)
        .build_session(
            "root",
            execution_mode,
            None::<StandardContextApproach>,
            None,
        )
        .context("build plugin session")
}

/// Locked-down tool surface visible to the model and inherited by every
/// `spawn_agent` child. Mirrors continual-learning-bench's tool list.
fn longcot_tool_definitions() -> Vec<ToolDefinition> {
    let capabilities = vec![SUBAGENT_CAPABILITY.to_string()];
    vec![
        lash_llm_tools::llm_query_tool_definition(),
        lash_mode_rlm::continue_as_tool_definition(),
        longcot_list_async_handles_tool_definition(),
        lash_subagents::spawn_agent_tool_definition(&capabilities),
    ]
}

/// Custom `list_async_handles` registration so the tool shows up on the model's
/// surface; the RLM session runtime intercepts the call and answers it without
/// reaching this provider's `execute`.
struct LongCoTAsyncHandlesTool;

#[async_trait]
impl ToolProvider for LongCoTAsyncHandlesTool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![longcot_list_async_handles_tool_definition()]
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        ToolResult::err_fmt(format_args!(
            "`{}` is handled by the RLM session runtime and cannot run directly",
            call.name
        ))
    }
}

fn longcot_list_async_handles_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "list_async_handles",
        "List live lashlang async handles only. Returns `{ monitor: { monitor_id: handle }, subagent: { name: handle }, tool: { id: handle } }`; terminal, awaited, or cancelled handles are omitted. Use this to rediscover live `start call` handles after a long-running fan-out via `spawn_agent`.",
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

/// Prompt template tuned for LongCoT. The benchmark problem is bound as
/// `input.prompt` in the Bound Variables preamble (see
/// `build_globals_patch`), so this template is deliberately short: just an
/// intro, a pointer to the bound input, and the slots lash needs for runtime
/// context and tool schemas.
/// LongCoT-specific prompt template. Keeps lash's load-bearing RLM scaffolding
/// (lashlang syntax guide via `PromptBuiltin::ExecutionInstructions` and the
/// `Bound Variables` plugin contribution via `PromptSlot::Guidance`) but swaps
/// the generic "AI coding assistant" intro for a LongCoT-specific one and drops
/// the coding-agent `CoreGuidance` that doesn't apply to pure-reasoning
/// problems.
/// Decomposition-focused template. The benchmark intentionally forbids code
/// execution or external solvers, but recursive self-calls via `spawn_agent`
/// are in-bounds — that's exactly the recursive strategy that takes the blog's
/// numbers from 2.6% → 45%. Each subagent gets a fresh context window, so a
/// 50k-token monolithic problem can be tackled as a chain of bounded 3-8k
/// sub-tasks. This template nudges the model toward that pattern rather than
/// trying to solve everything inline.
fn longcot_prompt_template() -> PromptTemplate {
    PromptTemplate::new(vec![
        PromptTemplateSection::untitled(vec![
            PromptTemplateEntry::text(
                "You are solving a LongCoT problem. The problem text is bound as `input.prompt` (possibly several thousand tokens). Inspect it programmatically rather than materializing the whole value when you can.",
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
            "Strategy",
            vec![PromptTemplateEntry::text(LONGCOT_DECOMPOSITION_GUIDANCE)],
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

const LONGCOT_DECOMPOSITION_GUIDANCE: &str = r#"This benchmark forbids external tools, solvers, and Python — but it does NOT forbid recursive self-calls. `spawn_agent` starts a child model call with its own fresh context window. Use it. Long-horizon problems rarely succeed as a single monolithic reasoning pass; they routinely succeed as 3–6 bounded sub-calls stitched together.

One capability is registered: `default`, which uses the same model and reasoning settings as the root turn. Every `spawn_agent` call must pass `capability: "default"`.

A child session starts blank — none of the parent's globals or projected bindings are inherited automatically. Pass everything the child needs through `seed: { name: value, ... }`. Each entry's kind is preserved by source: `seed: { problem: input.prompt }` makes `problem` a read-only projected binding on the child (identical to how `input.prompt` reads on the parent), while `seed: { findings: my_findings }` lands as a regular RLM global. Computed values (e.g. `seed: { hint: slice(input.prompt, 0, 1000) }`) default to global. Use seed for everything the child must read; use `task` only for the directive prose.

Default pattern (adjust to the domain):

1. Classify & extract. First `spawn_agent` receives `seed: { problem: input.prompt }` and returns a compact structured summary: domain (logic/chess/chemistry/cs/math), initial state, goal state, hard constraints. Do NOT solve at this step.
2. Plan. Second `spawn_agent` proposes a solution plan as a list of concrete, bounded sub-problems.
3. Execute sub-problems. One `spawn_agent` per sub-problem. Pass the relevant slice and any prior findings through `seed:`. Where the sub-problems are independent, dispatch them in parallel via `start`/`await` so child contexts don't accumulate. Keep each child task narrowly scoped (≤ ~3k tokens of output).
4. Stitch & verify. A final `spawn_agent` (or your own root reasoning turn) assembles the pieces and — critically — verifies the concrete end state before emitting `solution = <value>`.

Budget discipline: you have a 50-iteration root turn limit. One monolithic try that overflows context is worse than a tree of smaller calls. If you catch yourself reasoning line-by-line in prose over hundreds of items, stop and spawn a subagent for that sub-problem instead."#;

/// Bind the LongCoT problem as RLM globals under `input`. The RLM preamble
/// then surfaces an `Input` record type and a "Bound Variables" entry so the
/// model knows what's available without the prompt being duplicated in the
/// user message.
fn build_projected_bindings(
    question: &LongCoTQuestion,
) -> anyhow::Result<lash_mode_rlm::RlmProjectedBindings> {
    Ok(lash_mode_rlm::RlmProjectedBindings::new()
        .bind_json(
            "benchmark",
            json!({
                "name": "LongCoT",
                "question_id": question.question_id,
                "domain": question.domain,
                "difficulty": question.difficulty,
                "template": question.template,
            }),
        )?
        .bind_json(
            "input",
            json!({
                "prompt": question.prompt,
                "question_id": question.question_id,
                "domain": question.domain,
                "difficulty": question.difficulty,
                "template": question.template,
            }),
        )?)
}

fn resolve_provider(args: &Args) -> anyhow::Result<ProviderHandle> {
    match args.provider_id.as_str() {
        // The original env-driven path: OpenRouter / any OpenAI-compatible
        // endpoint with a bare API key. Keeps the no-config workflow intact.
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
        // Any other provider key (e.g. `codex`, `anthropic`) is resolved from
        // the user's `~/.lash/config.json`, mirroring the clbench runner. This
        // is what enables `--provider-id codex` to use OAuth tokens off the
        // active Codex subscription.
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
    }
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

fn non_empty(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn has_solution_line(text: &str) -> bool {
    use regex::Regex;
    static mut CACHE: Option<Regex> = None;
    // Safe: single-threaded compile after first access; Regex is Sync.
    let re = unsafe {
        #[allow(static_mut_refs)]
        CACHE.get_or_insert_with(|| Regex::new(r"(?m)^\s*solution\s*=").unwrap())
    };
    re.is_match(text)
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    let text = serde_json::to_string_pretty(value)?;
    fs::write(path, format!("{text}\n")).with_context(|| format!("write {}", path.display()))
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

fn write_trace_index(
    output_dir: &Path,
    run_id: &str,
    results: &[QuestionResult],
) -> anyhow::Result<()> {
    let rows: String = results
        .iter()
        .map(|r| {
            let qid = html_escape(&r.question_id);
            let domain = html_escape(&r.domain);
            let difficulty = html_escape(&r.difficulty);
            let status = html_escape(&r.status);
            let badge_class = if r.successful { "ok" } else { "fail" };
            let solution = if r.solution_line_present { "yes" } else { "no" };
            format!(
                "<tr>\
                   <td><a href=\"questions/{qid}/trace.html\">{qid}</a></td>\
                   <td>{domain}</td>\
                   <td>{difficulty}</td>\
                   <td class=\"{badge_class}\">{status}</td>\
                   <td>{iters}</td>\
                   <td>{seconds:.1}s</td>\
                   <td>{solution}</td>\
                   <td><a href=\"questions/{qid}/system_prompt.txt\">system</a> · \
                       <a href=\"questions/{qid}/prompt.txt\">prompt</a> · \
                       <a href=\"questions/{qid}/answer.txt\">answer</a> · \
                       <a href=\"questions/{qid}/events.jsonl\">events</a></td>\
                 </tr>",
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
<title>LongCoT run {run_id}</title>
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
<h1>LongCoT run <code>{run_id}</code></h1>
<p class="meta">{count} questions · see <a href="results.json">results.json</a> / <a href="manifest.json">manifest.json</a></p>
<table>
  <thead>
    <tr>
      <th>question_id</th><th>domain</th><th>difficulty</th><th>status</th>
      <th>iters</th><th>elapsed</th><th>solution line</th><th>artifacts</th>
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

fn aggregate_by_domain(results: &[QuestionResult]) -> BTreeMap<String, DomainBucket> {
    let mut out = BTreeMap::<String, DomainBucket>::new();
    for r in results {
        let bucket = out.entry(r.domain.clone()).or_default();
        bucket.count += 1;
        if r.successful {
            bucket.successful += 1;
        }
        if r.solution_line_present {
            bucket.solution_line_present += 1;
        }
    }
    out
}

struct LongCoTEventSink {
    file: Mutex<File>,
    last_llm_response: Mutex<Option<String>>,
    iteration_count: Mutex<BTreeSet<usize>>,
    last_error: Mutex<Option<String>>,
}

impl LongCoTEventSink {
    fn new(path: PathBuf) -> anyhow::Result<Self> {
        let file = File::create(&path).with_context(|| format!("create {}", path.display()))?;
        Ok(Self {
            file: Mutex::new(file),
            last_llm_response: Mutex::new(None),
            iteration_count: Mutex::new(BTreeSet::new()),
            last_error: Mutex::new(None),
        })
    }

    fn last_llm_response(&self) -> Option<String> {
        self.last_llm_response.lock().ok().and_then(|v| v.clone())
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
}

#[async_trait::async_trait]
impl EventSink for LongCoTEventSink {
    async fn emit(&self, event: SessionEvent) {
        if let SessionEvent::LlmRequest { mode_iteration, .. } = &event
            && let Ok(mut turns) = self.iteration_count.lock()
        {
            turns.insert(*mode_iteration);
        }
        if let SessionEvent::LlmResponse { content, .. } = &event
            && let Ok(mut last) = self.last_llm_response.lock()
        {
            *last = Some(content.trim().to_string());
        }
        if let SessionEvent::Error { message, .. } = &event
            && let Ok(mut last) = self.last_error.lock()
        {
            *last = Some(message.clone());
        }
        if let Ok(line) = serde_json::to_string(&event)
            && let Ok(mut file) = self.file.lock()
        {
            let _ = writeln!(file, "{line}");
        }
    }
}
