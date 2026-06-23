use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use clap::{Parser, Subcommand, ValueEnum};
use lash::runtime::ProtocolTurnOptions;
use lash::{RlmCore, TurnActivity, TurnEvent};
use lash_cli::config::LashConfig;
use lash_core::TurnInput;
use lash_harness_opt::clbench::{ClbenchConfig, ClbenchProject};
use lash_harness_opt::strategies::gepa::{
    ReflectiveGepaStrategy, ReflectiveProposalRequest, ReflectiveProposer,
};
use lash_harness_opt::{
    Candidate, CandidateSelection, ComponentSelection, FrontierMode, HarnessOptStore,
    HarnessOptimizer, HarnessProject, OptimizationConfig, OptimizationRun, ProjectHarnessRunner,
    Split, SqliteHarnessStore,
};
use lash_rlm_types::{RlmCreateExtras, RlmTermination};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

#[derive(Parser, Debug)]
#[command(name = "lash-harness-opt")]
#[command(about = "Optimize Lash harness projects with pluggable strategies.")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Optimize {
        #[command(subcommand)]
        project: OptimizeProject,
    },
    Check {
        #[command(subcommand)]
        project: CheckProject,
    },
    Stats {
        run_dir: PathBuf,
    },
    Eval {
        #[command(subcommand)]
        project: EvalProject,
    },
}

#[derive(Subcommand, Debug)]
enum OptimizeProject {
    Clbench {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        run_dir: PathBuf,
        #[arg(long, default_value_t = 1)]
        max_metric_calls: u64,
        #[arg(long, default_value_t = 4)]
        minibatch_size: usize,
        #[arg(long, default_value_t = 4)]
        max_concurrency: usize,
        #[arg(long, default_value_t = 1.0)]
        perfect_score: f64,
        #[arg(long, default_value_t = true)]
        skip_perfect_score: bool,
        #[arg(long, value_enum, default_value_t = CliCandidateSelection::Pareto)]
        candidate_selection: CliCandidateSelection,
        #[arg(long, value_enum, default_value_t = CliComponentSelection::RoundRobin)]
        component_selection: CliComponentSelection,
        #[arg(long, value_enum, default_value_t = CliFrontierMode::Instance)]
        frontier: CliFrontierMode,
        #[arg(long, default_value_t = false)]
        use_merge: bool,
        #[arg(long, default_value_t = 0)]
        max_merge_invocations: usize,
        #[arg(long)]
        provider_id: Option<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        variant: Option<String>,
        #[arg(long, default_value_t = 1_000_000)]
        proposer_max_context_tokens: usize,
        #[arg(long)]
        proposer_prompt: Option<PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
enum CheckProject {
    Clbench {
        #[arg(long)]
        config: PathBuf,
    },
}

#[derive(Subcommand, Debug)]
enum EvalProject {
    Clbench {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
        #[arg(long, value_enum)]
        split: CliSplit,
        #[arg(long)]
        run_dir: Option<PathBuf>,
    },
}

#[derive(Clone, Debug, ValueEnum)]
enum CliSplit {
    Train,
    Val,
    Test,
}

#[derive(Clone, Debug, ValueEnum)]
enum CliCandidateSelection {
    Pareto,
    CurrentBest,
    EpsilonGreedy,
}

impl From<CliCandidateSelection> for CandidateSelection {
    fn from(value: CliCandidateSelection) -> Self {
        match value {
            CliCandidateSelection::Pareto => CandidateSelection::Pareto,
            CliCandidateSelection::CurrentBest => CandidateSelection::CurrentBest,
            CliCandidateSelection::EpsilonGreedy => CandidateSelection::EpsilonGreedy,
        }
    }
}

#[derive(Clone, Debug, ValueEnum)]
enum CliComponentSelection {
    RoundRobin,
    All,
}

impl From<CliComponentSelection> for ComponentSelection {
    fn from(value: CliComponentSelection) -> Self {
        match value {
            CliComponentSelection::RoundRobin => ComponentSelection::RoundRobin,
            CliComponentSelection::All => ComponentSelection::All,
        }
    }
}

#[derive(Clone, Debug, ValueEnum)]
enum CliFrontierMode {
    Instance,
    Objective,
    Hybrid,
    Cartesian,
}

impl From<CliFrontierMode> for FrontierMode {
    fn from(value: CliFrontierMode) -> Self {
        match value {
            CliFrontierMode::Instance => FrontierMode::Instance,
            CliFrontierMode::Objective => FrontierMode::Objective,
            CliFrontierMode::Hybrid => FrontierMode::Hybrid,
            CliFrontierMode::Cartesian => FrontierMode::Cartesian,
        }
    }
}

impl From<CliSplit> for Split {
    fn from(value: CliSplit) -> Self {
        match value {
            CliSplit::Train => Split::Train,
            CliSplit::Val => Split::Val,
            CliSplit::Test => Split::Test,
        }
    }
}

const DEFAULT_TOKIO_THREAD_STACK_BYTES: usize = 2 * 1024 * 1024;

fn main() -> Result<()> {
    let stack_bytes = std::env::var("LASH_HARNESS_OPT_TOKIO_STACK_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_TOKIO_THREAD_STACK_BYTES);
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(stack_bytes)
        .build()
        .context("build lash-harness-opt Tokio runtime")?
        .block_on(async_main())
}

async fn async_main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Optimize { project } => match project {
            OptimizeProject::Clbench {
                config,
                run_dir,
                max_metric_calls,
                minibatch_size,
                max_concurrency,
                perfect_score,
                skip_perfect_score,
                candidate_selection,
                component_selection,
                frontier,
                use_merge,
                max_merge_invocations,
                provider_id,
                model,
                variant,
                proposer_max_context_tokens,
                proposer_prompt,
            } => {
                let project = Arc::new(load_clbench_config(config).await?);
                let train = project.trainset().await?;
                if train.is_empty() {
                    bail!("clbench config has no train examples");
                }
                let val = project.valset().await?;
                let provider = resolve_provider(provider_id.as_deref())?;
                let run = OptimizationRun {
                    run_id: uuid::Uuid::new_v4().to_string(),
                    experiment_id: "clbench".to_string(),
                    run_dir: run_dir.clone(),
                    config: OptimizationConfig {
                        max_metric_calls,
                        max_iterations: None,
                        minibatch_size,
                        max_concurrency,
                        perfect_score,
                        skip_perfect_score,
                        candidate_selection: candidate_selection.into(),
                        component_selection: component_selection.into(),
                        frontier: frontier.into(),
                        use_merge,
                        max_merge_invocations,
                        ..OptimizationConfig::default()
                    },
                };
                let proposer = LashRlmReflectiveProposer::new(
                    provider,
                    model,
                    variant,
                    proposer_max_context_tokens,
                    proposer_prompt,
                );
                let optimizer = HarnessOptimizer::new(
                    ProjectHarnessRunner::new(project.clone()),
                    ReflectiveGepaStrategy::new(proposer),
                );
                let store = SqliteHarnessStore::open(&run_dir).await?;
                let state = optimizer
                    .run_with_store(
                        run,
                        project.seed_candidate().await?,
                        train,
                        val,
                        &store,
                        CancellationToken::new(),
                    )
                    .await?;
                println!("{}", serde_json::to_string_pretty(&state)?);
            }
        },
        Command::Check { project } => match project {
            CheckProject::Clbench { config } => {
                let project = load_clbench_config(config).await?;
                let seed = project.seed_candidate().await?;
                let train = project.trainset().await?;
                let val = project.valset().await?;
                let test = project.testset().await?;
                let _ = ClbenchProject::prompt_template(&seed)?;
                let _ = ClbenchProject::user_directive(&seed)?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "status": "ok",
                        "project": "clbench",
                        "mutable_components": seed.mutable_components.keys().collect::<Vec<_>>(),
                        "examples": {
                            "train": train.len(),
                            "val": val.len(),
                            "test": test.len()
                        }
                    }))?
                );
            }
        },
        Command::Stats { run_dir } => {
            let store = SqliteHarnessStore::open(&run_dir).await?;
            let run = store.load_run().await?.ok_or_else(|| {
                anyhow::anyhow!("no harness-opt.sqlite run in {}", run_dir.display())
            })?;
            let state = lash_harness_opt::load_state(&run, &store).await?;
            let best = state.best();
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "run_id": state.run.run_id,
                    "experiment_id": state.run.experiment_id,
                    "evaluated_candidates": state.evaluated_candidates.len(),
                    "best_candidate_id": best.map(|best| best.candidate.id.clone()),
                    "best_mean_score": best.map(|best| best.mean_score()),
                    "metric_calls_used": state.metric_calls_used,
                    "accepted_proposals": state.accepted_proposals,
                    "rejected_proposals": state.rejected_proposals,
                    "cache_hits": state.cache_hits,
                    "cache_misses": state.cache_misses,
                    "pareto_coverage": lash_harness_opt::frontier(&state.evaluated_candidates).len(),
                }))?
            );
        }
        Command::Eval { project } => match project {
            EvalProject::Clbench {
                config,
                candidate,
                split,
                run_dir,
            } => {
                let project = Arc::new(load_clbench_config(config).await?);
                let candidate: Candidate = serde_json::from_slice(
                    &tokio::fs::read(&candidate)
                        .await
                        .with_context(|| format!("read {}", candidate.display()))?,
                )
                .with_context(|| format!("parse {}", candidate.display()))?;
                let examples = match Split::from(split) {
                    Split::Train => project.trainset().await?,
                    Split::Val => project.valset().await?,
                    Split::Test => project.testset().await?,
                };
                let run = OptimizationRun {
                    run_id: uuid::Uuid::new_v4().to_string(),
                    experiment_id: "clbench".to_string(),
                    run_dir: run_dir.unwrap_or_else(|| {
                        std::env::temp_dir()
                            .join(format!("lash-harness-opt-eval-{}", uuid::Uuid::new_v4()))
                    }),
                    config: OptimizationConfig {
                        max_metric_calls: examples.len() as u64,
                        max_iterations: Some(0),
                        minibatch_size: examples.len().max(1),
                        max_concurrency: 4,
                        per_example_timeout_secs: None,
                        ..OptimizationConfig::default()
                    },
                };
                let runner = ProjectHarnessRunner::new(project);
                let evaluation = lash_harness_opt::HarnessRunner::evaluate_candidate(
                    &runner,
                    &run,
                    candidate,
                    examples,
                    CancellationToken::new(),
                )
                .await?;
                println!("{}", serde_json::to_string_pretty(&evaluation)?);
            }
        },
    }
    Ok(())
}

async fn load_clbench_config(path: PathBuf) -> Result<ClbenchProject> {
    let bytes = tokio::fs::read(&path)
        .await
        .with_context(|| format!("read {}", path.display()))?;
    let config: ClbenchConfig =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    Ok(ClbenchProject::new(config))
}

struct LashRlmReflectiveProposer {
    provider: lash_core::ProviderHandle,
    model: Option<String>,
    variant: Option<String>,
    max_context_tokens: usize,
    prompt_template_path: Option<PathBuf>,
}

impl LashRlmReflectiveProposer {
    fn new(
        provider: lash_core::ProviderHandle,
        model: Option<String>,
        variant: Option<String>,
        max_context_tokens: usize,
        prompt_template_path: Option<PathBuf>,
    ) -> Self {
        Self {
            provider,
            model,
            variant,
            max_context_tokens,
            prompt_template_path,
        }
    }
}

#[async_trait]
impl ReflectiveProposer for LashRlmReflectiveProposer {
    async fn propose_json(
        &self,
        request: ReflectiveProposalRequest,
        cancellation: CancellationToken,
    ) -> lash_harness_opt::Result<Value> {
        let model = self
            .model
            .clone()
            .unwrap_or_else(|| default_model_for_provider(self.provider.kind()).to_string());
        let model_spec = lash::ModelSpec::from_token_limits(
            model,
            self.variant.clone(),
            self.max_context_tokens,
            None,
        )
        .map_err(|error| lash_harness_opt::HarnessOptError::Strategy(error.to_string()))?;
        let core_builder = RlmCore::builder()
            .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
            .lashlang_artifact_store(Arc::new(
                lash::persistence::InMemoryLashlangArtifactStore::new(),
            ))
            .attachment_store(Arc::new(lash::persistence::InMemoryAttachmentStore::new()))
            .provider(self.provider.clone())
            .model(model_spec);
        let core = core_builder
            .build()
            .map_err(|error| lash_harness_opt::HarnessOptError::Strategy(error.to_string()))?;
        let session = core
            .session(format!(
                "harness-opt-gepa-{}-{}",
                request.run_id, request.generation
            ))
            .open()
            .await
            .map_err(|error| lash_harness_opt::HarnessOptError::Strategy(error.to_string()))?;
        let prompt = render_gepa_proposer_prompt(&request, self.prompt_template_path.as_ref())
            .map_err(|error| {
                lash_harness_opt::HarnessOptError::Strategy(format!("render GEPA prompt: {error}"))
            })?;
        tokio::fs::create_dir_all(&request.artifact_dir)
            .await
            .map_err(lash_harness_opt::HarnessOptError::Io)?;
        let prompt_path = request.artifact_dir.join("prompt.md");
        tokio::fs::write(&prompt_path, &prompt)
            .await
            .map_err(lash_harness_opt::HarnessOptError::Io)?;
        let effect_host = session.effect_host();
        let scoped_effect_controller = effect_host
            .scoped(lash::runtime::ExecutionScope::turn(
                session.session_id(),
                format!(
                    "harness-opt-gepa-turn-{}-{}",
                    request.run_id, request.generation
                ),
            ))
            .map_err(|error| lash_harness_opt::HarnessOptError::Strategy(error.to_string()))?;
        let turn = session
            .turn(TurnInput::text(prompt))
            .cancel(cancellation)
            .protocol_turn_options(
                ProtocolTurnOptions::typed(RlmCreateExtras {
                    termination: RlmTermination::SubmitRequired {
                        schema: Some(request.output_schema.clone()),
                    },
                    final_answer_format: None,
                })
                .map_err(|error| lash_harness_opt::HarnessOptError::Strategy(error.to_string()))?,
            )
            .advanced()
            .run_with_scope(scoped_effect_controller)
            .await
            .map_err(|error| lash_harness_opt::HarnessOptError::Strategy(error.to_string()))?;

        let assistant_prose = assistant_prose(&turn.activities);
        match turn.result.outcome {
            lash_core::TurnOutcome::Finished(lash_core::TurnFinish::SubmittedValue { value })
            | lash_core::TurnOutcome::Finished(lash_core::TurnFinish::ToolValue {
                value, ..
            }) => {
                let output_path = request.artifact_dir.join("output.json");
                tokio::fs::write(
                    &output_path,
                    serde_json::to_vec_pretty(&json!({
                        "value": value,
                        "errors": turn.result.errors,
                        "assistant_prose": assistant_prose,
                    }))
                    .map_err(lash_harness_opt::HarnessOptError::Json)?,
                )
                .await
                .map_err(lash_harness_opt::HarnessOptError::Io)?;
                Ok(with_proposal_artifact_refs(
                    value,
                    &prompt_path,
                    &output_path,
                ))
            }
            other => Err(lash_harness_opt::HarnessOptError::Strategy(format!(
                "GEPA RLM proposer did not submit a proposal: outcome={other:?} errors={:?} output={}",
                turn.result.errors, assistant_prose
            ))),
        }
    }
}

fn default_model_for_provider(kind: &str) -> &'static str {
    match kind {
        "anthropic" => "claude-opus-4-7",
        "openai" => "gpt-5.4",
        "openai-compatible" => "anthropic/claude-sonnet-4.6",
        "codex" => "gpt-5.5",
        "google_oauth" => "gemini-3.1-pro-preview",
        _ => "mock-model",
    }
}

fn assistant_prose(activities: &[TurnActivity]) -> String {
    activities
        .iter()
        .filter_map(|activity| match &activity.event {
            TurnEvent::AssistantProseDelta { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

fn with_proposal_artifact_refs(mut value: Value, prompt_path: &Path, output_path: &Path) -> Value {
    let Some(proposals) = value.get_mut("proposals").and_then(Value::as_array_mut) else {
        return value;
    };
    for proposal in proposals {
        let Some(object) = proposal.as_object_mut() else {
            continue;
        };
        let metadata = object.entry("metadata").or_insert_with(|| json!({}));
        if let Some(metadata) = metadata.as_object_mut() {
            metadata.insert(
                "rlm_prompt_ref".to_string(),
                Value::String(prompt_path.to_string_lossy().to_string()),
            );
            metadata.insert(
                "rlm_output_ref".to_string(),
                Value::String(output_path.to_string_lossy().to_string()),
            );
        }
    }
    value
}

fn render_gepa_proposer_prompt(
    request: &ReflectiveProposalRequest,
    template_path: Option<&PathBuf>,
) -> Result<String> {
    let values = PromptTemplateValues {
        parent_candidate_id: request.parent_candidate_id.as_str(),
        mutable_components_json: serde_json::to_string_pretty(&request.mutable_components)?,
        evidence_json: serde_json::to_string_pretty(&request.evidence)?,
        output_schema_json: serde_json::to_string_pretty(&request.output_schema)?,
    };
    let template = if let Some(path) = template_path {
        std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?
    } else {
        default_gepa_proposer_prompt_template().to_string()
    };
    Ok(render_prompt_template(&template, &values))
}

struct PromptTemplateValues<'a> {
    parent_candidate_id: &'a str,
    mutable_components_json: String,
    evidence_json: String,
    output_schema_json: String,
}

fn render_prompt_template(template: &str, values: &PromptTemplateValues<'_>) -> String {
    template
        .replace("{{parent_candidate_id}}", values.parent_candidate_id)
        .replace(
            "{{mutable_components_json}}",
            &values.mutable_components_json,
        )
        .replace("{{evidence_json}}", &values.evidence_json)
        .replace("{{output_schema_json}}", &values.output_schema_json)
}

fn default_gepa_proposer_prompt_template() -> &'static str {
    r#"You are GEPA inside Lash RLM mode. Reflect on the evaluation evidence and propose concrete candidate patches for the harness optimizer.

Parent candidate id:
{{parent_candidate_id}}

Mutable components:
{{mutable_components_json}}

Evaluation evidence:
{{evidence_json}}

Required output schema:
{{output_schema_json}}

Rules:
- Propose patches only for listed mutable component ids.
- Preserve generic RLM execution protocol, Lashlang reference, tool contracts, Tool Catalog membership, and typed response-schema enforcement.
- Use feedback and traces to target the weakest selected component.
- Prefer small, testable changes over broad rewrites.
- Do not claim benchmark improvements that are not present in the evidence.

Return only by calling `submit <object>` from a single paired `<lashlang>` block closed with `</lashlang>`. The submitted object must match the required output schema."#
}

fn resolve_provider(provider_id: Option<&str>) -> Result<lash_core::ProviderHandle> {
    let config_path = lash_home().join("config.json");
    let mut config = LashConfig::load(&config_path)
        .ok_or_else(|| anyhow::anyhow!("missing or invalid {}", config_path.display()))?;
    if let Some(provider_id) = provider_id {
        config
            .set_active_provider_kind(provider_id)
            .map_err(anyhow::Error::msg)?;
    }
    config.build_active_provider().map_err(anyhow::Error::msg)
}

fn lash_home() -> PathBuf {
    std::env::var_os("LASH_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".lash")))
        .unwrap_or_else(|| PathBuf::from(".lash"))
}
