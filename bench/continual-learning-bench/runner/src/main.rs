use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::Parser;
use lash::plugin::PluginFactory;
use lash::provider::LashConfig;
use lash::{
    AppendSessionNodesRequest, BackgroundRuntimeHost, BuiltinToolResultProjectionPluginFactory,
    EmbeddedRuntimeHost, EventSink, ExecutionMode, InputItem, LashRuntime, NoopEventSink,
    PersistedSessionState, PersistentRuntimeServices, PluginHost, PromptBuiltin, PromptSlot,
    PromptTemplate, PromptTemplateEntry, PromptTemplateSection, ProviderHandle, RuntimeCoreConfig,
    RuntimePersistence, SessionAppendNode, SessionEventRecord, SessionPolicy, SessionUsageReport,
    StandardContextApproach, TokioSessionTaskExecutor, TurnFinish, TurnInjectionBridge, TurnInput,
    TurnInputInjectionBridge, TurnOutcome, diff_usage_reports,
};
use lash_plugin_rolling_history::RollingHistoryPluginFactory;
use lash_rlm_types::{RlmGlobalsPatchPluginBody, RlmModeEvent, RlmTermination};
use lash_sqlite_store::Store;
use lash_subagents::{
    CapabilityField, CapabilityOptionalField, CapabilityRecursion, CapabilityRegistry,
    CapabilitySpec, CapabilityToolSurface, LocalSubagentHost, StaticCapability, SubagentHost,
    SubagentsPluginFactory,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const DEFAULT_MAX_CONTEXT_TOKENS: usize = 1_000_000;
const DEFAULT_MAX_TURNS: usize = 30;

#[derive(Parser, Debug)]
#[command(name = "bench-clbench-lash")]
#[command(about = "Run one Continual Learning Bench query through Lash RLM.")]
struct Args {
    #[arg(long)]
    request: PathBuf,
}

#[derive(Debug, Deserialize)]
struct RunnerRequest {
    session_id: String,
    session_db: PathBuf,
    trace_path: Option<PathBuf>,
    model: String,
    provider_id: Option<String>,
    variant: Option<String>,
    max_context_tokens: Option<usize>,
    max_turns: Option<usize>,
    iteration: usize,
    prompt: String,
    feedback: Option<String>,
    response_schema: Value,
    init_diary: bool,
}

#[derive(Debug, Serialize)]
struct RunnerResponse {
    action: Value,
    session_id: String,
    status: String,
    done_reason: String,
    usage: Value,
    diagnostics: Value,
}

#[tokio::main]
async fn main() -> Result<()> {
    lash_providers_builtin::register_all();
    let args = Args::parse();
    let request: RunnerRequest = serde_json::from_str(
        &fs::read_to_string(&args.request)
            .with_context(|| format!("read {}", args.request.display()))?,
    )
    .with_context(|| format!("parse {}", args.request.display()))?;

    let response = run_query(request).await?;
    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}

async fn run_query(request: RunnerRequest) -> Result<RunnerResponse> {
    if let Some(parent) = request.session_db.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    if let Some(trace_path) = &request.trace_path
        && let Some(parent) = trace_path.parent()
    {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }

    let provider = resolve_provider(request.provider_id.as_deref())?;
    let execution_mode = ExecutionMode::new("rlm");
    let standard_context_approach = None;
    let policy = SessionPolicy {
        model: request.model.clone(),
        provider,
        model_variant: request.variant.clone(),
        max_context_tokens: Some(
            request
                .max_context_tokens
                .unwrap_or(DEFAULT_MAX_CONTEXT_TOKENS),
        ),
        max_turns: Some(request.max_turns.unwrap_or(DEFAULT_MAX_TURNS)),
        execution_mode: execution_mode.clone(),
        standard_context_approach: standard_context_approach.clone(),
        session_id: Some(request.session_id.clone()),
        ..SessionPolicy::default()
    };

    let store = Arc::new(
        Store::open(&request.session_db)
            .with_context(|| format!("open {}", request.session_db.display()))?,
    );
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
                .with_trace_jsonl_path(request.trace_path.clone())
                .with_prompt_template(clbench_prompt_template()),
        ),
        Arc::new(TokioSessionTaskExecutor::default()),
    );
    let state = lash::load_persisted_session_state(store.as_ref())
        .await
        .context("load session state")?
        .unwrap_or_else(|| PersistedSessionState {
            session_id: request.session_id.clone(),
            policy: policy.clone(),
            ..PersistedSessionState::default()
        });
    let mut runtime = LashRuntime::from_persistent_background_state(policy, host, services, state)
        .await
        .context("open runtime")?;

    runtime
        .append_session_nodes(AppendSessionNodesRequest {
            nodes: vec![SessionAppendNode::event(SessionEventRecord::Mode(
                lash::ModeEvent::rlm(RlmModeEvent::RlmGlobalsPatch(build_globals_patch(&request))),
            ))],
            requires_ancestor_node_id: None,
        })
        .await
        .context("bind clbench globals")?;

    let before_usage = runtime.usage_report();
    let sink = NoopEventSink;
    let turn = runtime
        .stream_turn(
            TurnInput {
                items: vec![InputItem::Text {
                    text: build_user_message(&request),
                }],
                image_blobs: Default::default(),
                user_input: None,
                mode: None,
                mode_turn_options: Some(lash::ModeTurnOptions::typed(
                    ExecutionMode::new("rlm"),
                    RlmTermination::Finish {
                        schema: Some(request.response_schema.clone()),
                        include_submit_prompt: true,
                    },
                )?),
            },
            &sink as &dyn EventSink,
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .context("run clbench turn")?;
    let after_usage = runtime.usage_report();
    let usage_rows = diff_usage_reports(&before_usage, &after_usage)
        .map_err(anyhow::Error::msg)
        .context("diff usage reports")?;
    let usage = SessionUsageReport::from_entries(&usage_rows);

    let action = match &turn.outcome {
        TurnOutcome::Finished(TurnFinish::Submission { value, .. }) => value.clone(),
        other => bail!(
            "turn did not submit an action: status={} reason={} errors={:?} output={}",
            turn_status_label(other),
            done_reason_label(other),
            turn.errors,
            turn.assistant_output.safe_text
        ),
    };

    Ok(RunnerResponse {
        action,
        session_id: request.session_id,
        status: turn_status_label(&turn.outcome).to_string(),
        done_reason: done_reason_label(&turn.outcome).to_string(),
        usage: serde_json::to_value(&usage)?,
        diagnostics: json!({
            "assistant_output_chars": turn.assistant_output.safe_text.chars().count(),
            "tool_call_count": turn.tool_calls.len(),
            "error_count": turn.errors.len(),
        }),
    })
}

fn build_plugin_session(
    execution_mode: ExecutionMode,
    policy: &SessionPolicy,
) -> Result<Arc<lash::PluginSession>> {
    let factories: Vec<Arc<dyn PluginFactory>> = vec![
        Arc::new(BuiltinToolResultProjectionPluginFactory::default()),
        Arc::new(RollingHistoryPluginFactory::default()),
        Arc::new(lash_mode_rlm::BuiltinRlmModePluginFactory::default()),
        Arc::new(SubagentsPluginFactory::new(
            policy.clone(),
            Arc::new(
                CapabilityRegistry::new().with(Arc::new(StaticCapability::new(
                    "explore",
                    CapabilitySpec {
                        model: CapabilityField::Inherit,
                        model_variant: CapabilityOptionalField::Inherit,
                        execution_mode: CapabilityField::Inherit,
                        tool_surface: CapabilityToolSurface::CognitionOnly,
                        recursion: CapabilityRecursion::Disabled,
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

fn clbench_prompt_template() -> PromptTemplate {
    PromptTemplate::new(vec![
        PromptTemplateSection::untitled(vec![PromptTemplateEntry::text(
            "You are being evaluated by Continual Learning Bench, which tests whether an agent improves from feedback across sequential task instances.",
        )]),
        PromptTemplateSection::titled(
            "Execution",
            vec![
                PromptTemplateEntry::builtin(PromptBuiltin::ExecutionInstructions),
                PromptTemplateEntry::slot(PromptSlot::Execution),
            ],
        ),
        PromptTemplateSection::titled(
            "Continual Memory",
            vec![PromptTemplateEntry::text(CLBENCH_MEMORY_GUIDANCE)],
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

const CLBENCH_MEMORY_GUIDANCE: &str = r#"Use your persistent Lash RLM REPL as memory. A global list named `diary` is already bound. Each entry is a short record:

`{ index: int, history_index: int, summary: str, learnings: str }`

The current benchmark iteration is bound as `iteration`.

At the start of each turn:
- If `diary` is small, inspect it directly.
- If `diary` is large, use parallel `llm_query` calls over diary entries or chunks to identify entries relevant to the current query and feedback.
- For relevant entries, inspect the matching `history[entry.history_index]` when more detail is needed.
- You may use `spawn_agent` with `capability: "explore"` to analyze one selected history entry or a small selected group of history entries and return focused lessons.
- Apply useful prior learnings before choosing the benchmark action.

Before every `submit`, append exactly one diary record:

```lashlang
diary = push(diary, {
  index: len(diary),
  history_index: len(history) - 1,
  summary: "brief task/action summary",
  learnings: "reusable lesson from this interaction"
})
submit { action: "..." }
```

Keep entries short, factual, and reusable. Do not duplicate old lessons; incorporate feedback and revise strategy in later entries. Use only the available cognition tools: `llm_query`, `spawn_agent` with `capability: "explore"`, `continue_as`, and async-handle helpers if needed. Do not use local shell, file, search, edit, web, or external tools.

Return the benchmark action by calling `submit <value>` from a fenced `lashlang` block. The submitted value must match the current response schema exactly."#;

fn build_globals_patch(request: &RunnerRequest) -> RlmGlobalsPatchPluginBody {
    let mut set = serde_json::Map::new();
    if request.init_diary {
        set.insert("diary".to_string(), Value::Array(Vec::new()));
    }
    set.insert("iteration".to_string(), json!(request.iteration));
    set.insert("current_query".to_string(), json!(request.prompt));
    set.insert(
        "current_feedback".to_string(),
        request
            .feedback
            .as_ref()
            .map(|feedback| json!(feedback))
            .unwrap_or(Value::Null),
    );
    set.insert(
        "benchmark_response_schema".to_string(),
        request.response_schema.clone(),
    );
    RlmGlobalsPatchPluginBody {
        set,
        unset: Vec::new(),
    }
}

fn build_user_message(request: &RunnerRequest) -> String {
    let mut parts = vec![format!("Benchmark iteration: {}", request.iteration)];
    if let Some(feedback) = request
        .feedback
        .as_deref()
        .filter(|text| !text.trim().is_empty())
    {
        parts.push(format!("Feedback from the previous action:\n{feedback}"));
    }
    parts.push(format!("Current query:\n{}", request.prompt));
    parts.push(
        "Choose the next benchmark action. Return it with `submit <value>` from a fenced `lashlang` block."
            .to_string(),
    );
    parts.join("\n\n")
}

fn resolve_provider(provider_id: Option<&str>) -> Result<ProviderHandle> {
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
        .unwrap_or_else(|| Path::new(".lash").to_path_buf())
}

fn turn_status_label(outcome: &TurnOutcome) -> &'static str {
    match outcome {
        TurnOutcome::Finished(_) | TurnOutcome::Handoff { .. } => "completed",
        TurnOutcome::Stopped(lash::TurnStop::Cancelled) => "interrupted",
        TurnOutcome::Stopped(_) => "failed",
    }
}

fn done_reason_label(outcome: &TurnOutcome) -> &'static str {
    match outcome {
        TurnOutcome::Finished(TurnFinish::AssistantMessage { .. }) => "assistant_message",
        TurnOutcome::Finished(TurnFinish::Submission { .. }) => "submission",
        TurnOutcome::Handoff { .. } => "handoff",
        TurnOutcome::Stopped(lash::TurnStop::Cancelled) => "cancelled",
        TurnOutcome::Stopped(lash::TurnStop::InvalidInput) => "invalid_input",
        TurnOutcome::Stopped(lash::TurnStop::MaxTurns) => "max_turns",
        TurnOutcome::Stopped(lash::TurnStop::ToolFailure) => "tool_failure",
        TurnOutcome::Stopped(lash::TurnStop::ProviderError) => "provider_error",
        TurnOutcome::Stopped(lash::TurnStop::PluginAbort) => "plugin_abort",
        TurnOutcome::Stopped(lash::TurnStop::RuntimeError) => "runtime_error",
        TurnOutcome::Stopped(lash::TurnStop::SubmittedError { .. }) => "submitted_error",
    }
}
