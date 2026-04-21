mod dataset;

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, bail};
use chrono::Utc;
use clap::Parser;
use dataset::{LongCoTQuestion, load_questions};
use lash::plugin::PluginFactory;
use lash::provider::OPENROUTER_BASE_URL;
use lash::{
    AppendSessionNodesRequest, BackgroundRuntimeHost, BuiltinToolResultProjectionPluginFactory,
    ContextApproach, EmbeddedRuntimeHost, EventSink, ExecutionMode, InputItem, LashRuntime,
    PersistedSessionState, PersistentRuntimeServices, PluginHost, PromptBuiltin, PromptSlot,
    PromptTemplate, PromptTemplateEntry, PromptTemplateSection, Provider, ProviderOptions,
    RlmGlobalsPatchPluginBody, RuntimeCoreConfig, RuntimeStore, SessionAppendNode, SessionEvent,
    SessionPolicy, SessionUsageReport, Store, TokioSessionTaskExecutor, TurnInjectionBridge,
    TurnInput, TurnInputInjectionBridge, diff_usage_reports,
};
use lash_export::{ExportFormat, SessionSelector, export};
use lash_plugin_observational_memory::ObservationalMemoryPluginFactory;
use lash_plugin_rolling_history::RollingHistoryPluginFactory;
use lash_subagents::{
    CapabilityRegistry, LocalSubagentHost, SubagentHost, SubagentsPluginFactory, TierCapability,
    TierExecutionMode,
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
    "End with plain prose (no fenced block) whose final line is exactly `solution = <value>` matching the shape the problem specifies; emit nothing after that line."
);

// Defaults. GPT-5.2 (`openai/gpt-5.2` via OpenRouter's OpenAI-
// compatible endpoint) is the working target. Model knobs mirror the
// upstream `src/configs/oai_gpt52.yaml`: `reasoning.effort=high` and
// `max_output_tokens=125000`. Matching those keeps our numbers
// directly comparable to the published leaderboard. `max_turns=50`
// still lines up with the RLM blog's dspy.RLM iteration cap; the
// execution engine is always lash's `lashlang`-backed RLM.
const DEFAULT_MODEL: &str = "openai/gpt-5.2";
const DEFAULT_VARIANT: &str = "high";
const DEFAULT_MAX_TURNS: usize = 50;
const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 125_000;
const DEFAULT_MAX_CONTEXT_TOKENS: usize = 1_000_000;
const DEFAULT_BATCH_SIZE: usize = 4;
const DEFAULT_EXECUTION_MODE: &str = "rlm";
const DEFAULT_CONTEXT_APPROACH: &str = "rolling_history";
const DEFAULT_HARNESS: &str = "restricted";

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
    #[arg(long, default_value = DEFAULT_EXECUTION_MODE)]
    execution_mode: String,

    #[arg(long, default_value = DEFAULT_CONTEXT_APPROACH)]
    context_approach: String,

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
    context_approach: String,
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
            reference_framework: "dspy.RLM v3.1.3".to_string(),
            note:
                "This harness runs LongCoT through lash's lashlang-backed RLM. The iteration cap \
                 (50) and max-output cap (64k) match the reference writeup; the execution engine \
                 is lash, not dspy.RLM."
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
    context_approach: String,
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

    let execution_mode = parse_execution_mode(&args.execution_mode)?;
    let context_approach = parse_context_approach(&args.context_approach)?;
    let model_slug = args.model.replace('/', "_").replace(':', "_");
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
        execution_mode: execution_mode_label(execution_mode).to_string(),
        context_approach: context_approach_label(&context_approach).to_string(),
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
    eprintln!("  context-approach: {}", manifest.context_approach);
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
        let execution_mode = execution_mode;
        let context_approach = context_approach.clone();
        join_set.spawn(async move {
            let _permit = permit;
            let result = run_question(
                output_dir.as_ref(),
                provider.as_ref(),
                args.as_ref(),
                execution_mode,
                &context_approach,
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
    provider: &Provider,
    args: &Args,
    execution_mode: ExecutionMode,
    context_approach: &ContextApproach,
    question: LongCoTQuestion,
) -> anyhow::Result<QuestionResult> {
    let question_dir = output_dir.join("questions").join(&question.question_id);
    fs::create_dir_all(&question_dir)
        .with_context(|| format!("create {}", question_dir.display()))?;
    write_json(&question_dir.join("question.json"), &question)?;
    fs::write(question_dir.join("prompt.txt"), &question.prompt)
        .with_context(|| format!("write {}", question_dir.join("prompt.txt").display()))?;

    let store_path = question_dir.join("session.db");
    let llm_log_path = question_dir.join("session.llm.jsonl");
    let store = Arc::new(
        Store::open(&store_path).with_context(|| format!("open {}", store_path.display()))?,
    );

    let policy = SessionPolicy {
        model: args.model.clone(),
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
                .with_llm_log_path(Some(llm_log_path.clone()))
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

    // Bind the problem as an RLM global so it sits in structured state rather
    // than the message body. The model sees an `input = { prompt, ... }` entry
    // in its Bound Variables preamble and can `observe input.prompt` from
    // inside a lashlang block, which is the lash analog of dspy's
    // `LongCoTSolve.prompt: dspy.InputField(...)`.
    runtime
        .append_session_nodes(AppendSessionNodesRequest {
            nodes: vec![SessionAppendNode::plugin(
                lash::INTERNAL_RLM_GLOBALS_PATCH_PLUGIN_TYPE,
                serde_json::to_value(build_globals_patch(&question))?,
            )],
            requires_ancestor_node_id: None,
        })
        .await
        .context("bind longcot input globals")?;

    let before_usage = runtime.usage_report();
    let started = std::time::Instant::now();
    let cancel = tokio_util::sync::CancellationToken::new();
    let sink = Arc::new(LongCoTEventSink::new(question_dir.join("events.jsonl"))?);
    let sink_trait: Arc<dyn EventSink> = sink.clone();
    let turn = runtime
        .stream_turn(
            TurnInput {
                items: vec![InputItem::Text {
                    text: LONGCOT_USER_DIRECTIVE.to_string(),
                }],
                image_blobs: Default::default(),
                user_input: None,
                mode: None,
                rlm_termination_override: None,
            },
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
    let status = turn_status_label(&turn.status).to_string();
    let done_reason = done_reason_label(&turn.done_reason).to_string();
    let successful = matches!(turn.status, lash::TurnStatus::Completed);
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
            execution_mode: execution_mode_label(execution_mode).to_string(),
            context_approach: context_approach_label(context_approach).to_string(),
            variant: Some(args.variant.clone()),
            max_turns: args.max_turns,
            max_output_tokens: args.max_output_tokens,
        },
    };
    write_json(&question_dir.join("result.json"), &result)?;

    // Emit a self-contained HTML trace alongside the session db. Failures
    // here should not take down the benchmark — traces are a debugging aid.
    let trace_path = question_dir.join("trace.html");
    if let Err(err) = export(
        SessionSelector::Path(&store_path),
        std::path::Path::new(""),
        ExportFormat::Html,
        Some(&trace_path),
    ) {
        eprintln!(
            "warn: failed to render HTML trace for {}: {err:#}",
            question.question_id
        );
    }

    // Project the actual outgoing system prompt out of the LLM log so you can
    // see exactly what the model was told. `lash-export` doesn't include the
    // on-the-fly rendered system prompt in the trace because it isn't stored
    // in the session graph.
    if let Err(err) = write_system_prompt_snapshot(&llm_log_path, &question_dir) {
        eprintln!(
            "warn: failed to snapshot system prompt for {}: {err:#}",
            question.question_id
        );
    }

    Ok(result)
}

fn write_system_prompt_snapshot(llm_log_path: &Path, question_dir: &Path) -> anyhow::Result<()> {
    if !llm_log_path.exists() {
        return Ok(());
    }
    let raw = fs::read_to_string(llm_log_path)
        .with_context(|| format!("read {}", llm_log_path.display()))?;
    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(record) = serde_json::from_str::<Value>(line) else {
            continue;
        };
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
            let text = extract_text(system.get("content"));
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

/// Build a minimal plugin stack: rolling-history context, tool-result
/// projection, and subagents (so RLM mode can spawn sub-agents for recursive
/// decomposition — the closest analogue to dspy.RLM). No benchmark-specific
/// tools: LongCoT prompts explicitly forbid external tool use.
fn build_plugin_session(
    execution_mode: ExecutionMode,
    context_approach: ContextApproach,
    policy: &SessionPolicy,
) -> anyhow::Result<Arc<lash::PluginSession>> {
    let mut factories: Vec<Arc<dyn PluginFactory>> =
        vec![Arc::new(BuiltinToolResultProjectionPluginFactory::default())];
    match context_approach {
        ContextApproach::RollingHistory(_) => {
            factories.push(Arc::new(RollingHistoryPluginFactory::default()));
        }
        ContextApproach::ObservationalMemory(_) => {
            factories.push(Arc::new(ObservationalMemoryPluginFactory));
        }
    }
    // The RLM runtime refuses to start without mode-session plugins
    // registered, so wire up both built-ins the same way lash-cli does.
    factories.push(Arc::new(
        lash_mode_standard::BuiltinStandardModePluginFactory,
    ));
    factories.push(Arc::new(
        lash_mode_rlm::BuiltinRlmModePluginFactory::default(),
    ));
    // Single capability `default` that inherits the root session's
    // model, variant, and execution mode. We deliberately drop the
    // `low` / `medium` / `high` tiers: the benchmark should only use
    // one model so subagent fanout is an amplification of the same
    // frontier model, not a quality-tiered delegation. An empty model
    // override on `TierCapability` falls back to the parent policy's
    // model via `pick_tier_model`.
    let registry =
        std::sync::Arc::new(CapabilityRegistry::new().with(Arc::new(TierCapability::new(
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
/// are in-bounds — that's exactly the dspy.RLM strategy that takes the blog's
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

Default pattern (adjust to the domain):

1. Classify & extract. First `spawn_agent` reads `input.prompt` and returns a compact structured summary: domain (logic/chess/chemistry/cs/math), initial state, goal state, hard constraints. Do NOT solve at this step.
2. Plan. Second `spawn_agent` proposes a solution plan as a list of concrete, bounded sub-problems.
3. Execute sub-problems. One `spawn_agent` per sub-problem. Where the sub-problems are independent, dispatch them in parallel via `start`/`await` so child contexts don't accumulate. Keep each child task narrowly scoped (≤ ~3k tokens of output) — that's the whole reason to decompose.
4. Stitch & verify. A final `spawn_agent` (or your own root reasoning turn) assembles the pieces and — critically — verifies the concrete end state before emitting `solution = <value>`.

Subagents inherit the bound variables, so they can read `input.prompt` directly (or specific slices you pass in via their `task` field). Keep each `task` description self-contained: the child has no memory of why you spawned it.

Budget discipline: you have a 50-iteration root turn limit. One monolithic try that overflows context is worse than a tree of smaller calls. If you catch yourself reasoning line-by-line in prose over hundreds of items, stop and spawn a subagent for that sub-problem instead."#;

/// Bind the LongCoT problem as RLM globals under `input`. The RLM preamble
/// then surfaces an `Input` record type and a "Bound Variables" entry so the
/// model knows what's available without the prompt being duplicated in the
/// user message.
fn build_globals_patch(question: &LongCoTQuestion) -> RlmGlobalsPatchPluginBody {
    let mut set = serde_json::Map::new();
    set.insert(
        "benchmark".to_string(),
        json!({
            "name": "LongCoT",
            "question_id": question.question_id,
            "domain": question.domain,
            "difficulty": question.difficulty,
            "template": question.template,
        }),
    );
    set.insert(
        "input".to_string(),
        json!({
            "prompt": question.prompt,
            "question_id": question.question_id,
            "domain": question.domain,
            "difficulty": question.difficulty,
            "template": question.template,
        }),
    );
    RlmGlobalsPatchPluginBody {
        set,
        unset: Vec::new(),
    }
}

fn resolve_provider(args: &Args) -> anyhow::Result<Provider> {
    match args.provider_id.as_str() {
        "openai-compatible" | "openai-generic" => {
            let api_key = resolve_api_key(args).ok_or_else(|| {
                anyhow::anyhow!(
                    "missing API key — set OPENROUTER_API_KEY or OPENAI_COMPATIBLE_API_KEY in .env, or pass --api-key"
                )
            })?;
            Ok(Provider::OpenAiGeneric {
                api_key,
                base_url: resolve_base_url(args),
                options: ProviderOptions::default(),
            })
        }
        other => bail!(
            "provider `{other}` is not supported by this harness; use the OpenAI-compatible path"
        ),
    }
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

fn parse_execution_mode(raw: &str) -> anyhow::Result<ExecutionMode> {
    match raw {
        "rlm" => Ok(ExecutionMode::Rlm),
        "standard" => Ok(ExecutionMode::Standard),
        _ => bail!("unsupported execution mode `{raw}`"),
    }
}

fn parse_context_approach(raw: &str) -> anyhow::Result<ContextApproach> {
    match raw {
        "rolling_history" => Ok(ContextApproach::RollingHistory(Default::default())),
        "observational_memory" => Ok(ContextApproach::ObservationalMemory(Default::default())),
        _ => bail!("unsupported context approach `{raw}`"),
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

fn turn_status_label(status: &lash::TurnStatus) -> &'static str {
    match status {
        lash::TurnStatus::Completed => "completed",
        lash::TurnStatus::Interrupted => "interrupted",
        lash::TurnStatus::Failed => "failed",
    }
}

fn done_reason_label(reason: &lash::DoneReason) -> &'static str {
    match reason {
        lash::DoneReason::ModelStop => "model_stop",
        lash::DoneReason::MaxTurns => "max_turns",
        lash::DoneReason::UserAbort => "user_abort",
        lash::DoneReason::ToolFailure => "tool_failure",
        lash::DoneReason::RuntimeError => "runtime_error",
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
        if let SessionEvent::LlmRequest { iteration, .. } = &event
            && let Ok(mut turns) = self.iteration_count.lock()
        {
            turns.insert(*iteration);
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
