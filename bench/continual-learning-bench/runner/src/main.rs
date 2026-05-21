use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use clap::Parser;
use lash::{
    LashCore, ModeId, ModePreset, PluginStack, SessionSpec, TurnInput,
    advanced::{EventSink, LocalProcessRegistry, TurnFinish, TurnOutcome, TurnStop},
    persistence::{
        ModeEvent, RuntimeSessionState, RuntimePersistence, load_persisted_session_state,
    },
    plugins::{PluginSpec, StaticPluginFactory},
    provider::ProviderHandle,
    tools::{
        ToolCall, ToolContract, ToolDefinition, ToolExecutionMode, ToolManifest, ToolProvider,
        ToolResult,
    },
    usage::{SessionUsageReport, TokenLedgerEntry},
};
use lash_cli::config::LashConfig;
use lash_harness_opt::clbench::CLBENCH_MEMORY_GUIDANCE;
use lash_llm_tools::LlmToolsPluginFactory;
use lash_mode_rlm::RlmTurnInputExt;
use lash_rlm_types::RlmModeEvent;
use lash_sqlite_store::Store;
use lash_subagents::{CapabilityRegistry, StaticCapability, SubagentsPluginFactory};
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
    let core = LashCore::builder()
        .install_mode(ModePreset::rlm_with_config(clbench_rlm_config()))
        .default_mode(ModeId::rlm())
        .provider(provider)
        .model(request.model.clone(), request.variant.clone())
        .max_context_tokens(
            request
                .max_context_tokens
                .unwrap_or(DEFAULT_MAX_CONTEXT_TOKENS),
        )
        .max_turns(request.max_turns.unwrap_or(DEFAULT_MAX_TURNS))
        .prompt_template(lash_harness_opt::clbench::clbench_prompt_template(
            CLBENCH_MEMORY_GUIDANCE,
        ))
        .trace_jsonl_path(request.trace_path.clone())
        .process_registry(Arc::new(LocalProcessRegistry::default()))
        .plugins(build_plugin_stack())
        .build()?;

    let store = Arc::new(
        Store::open(&request.session_db)
            .with_context(|| format!("open {}", request.session_db.display()))?,
    );
    let mut state = load_persisted_session_state(store.as_ref())
        .await
        .context("load session state")?
        .unwrap_or_else(|| RuntimeSessionState {
            session_id: request.session_id.clone(),
            ..RuntimeSessionState::default()
        });
    if let Some(seed) = build_seed_event(&request) {
        state.session_graph.append_mode_event(seed);
    }
    let session = core
        .session(request.session_id.clone())
        .rlm()
        .store(store.clone() as Arc<dyn RuntimePersistence>)
        .open_with_state(state)
        .await
        .context("open clbench session")?;

    let sink = lash::advanced::NoopEventSink;
    let mut turn_input = TurnInput::text(clbench_turn_text(&request))
        .rlm_project(build_projected_bindings(&request)?)?;
    turn_input.trace_turn_id = Some(format!("clbench-turn-{:04}", request.iteration));
    let followed = session
        .turn(turn_input)
        .require_submit_schema(request.response_schema.clone())?
        .collect_followed_session_events_with(&sink as &dyn EventSink)
        .await
        .context("run clbench turn")?;
    let usage = usage_from_followed_turn(&followed);
    let turn = followed
        .final_turn()
        .context("handoff chain did not produce a turn")?;

    let action = match &turn.outcome {
        TurnOutcome::Finished(TurnFinish::SubmittedValue { value })
        | TurnOutcome::Finished(TurnFinish::ToolValue { value, .. }) => value.clone(),
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
            "followed_turn_count": followed.turns.len(),
            "handoff_count": followed.handoff_count(),
        }),
    })
}

fn usage_from_followed_turn(followed: &lash::FollowedTurnResult) -> SessionUsageReport {
    let entries = followed
        .turns
        .iter()
        .filter(|turn| turn.usage.total() != 0 || turn.usage.cached_input_tokens != 0)
        .map(|turn| TokenLedgerEntry {
            source: "turn".to_string(),
            model: turn.state.policy.model.clone(),
            usage: turn.usage.clone(),
        })
        .collect::<Vec<_>>();
    SessionUsageReport::from_entries(&entries)
}

fn clbench_turn_text(request: &RunnerRequest) -> String {
    let mut text = String::from("Choose the next benchmark action.");
    text.push_str("\n\n=== CURRENT QUERY ===\n\n");
    text.push_str(&request.prompt);
    text.push_str("\n\n=== CURRENT FEEDBACK ===\n\n");
    match request.feedback.as_deref() {
        Some(feedback) if !feedback.trim().is_empty() => text.push_str(feedback),
        _ => text.push_str("null"),
    }
    text
}

fn clbench_rlm_config() -> lash_mode_rlm::RlmModePluginConfig {
    lash_mode_rlm::RlmModePluginConfig {
        prompt_features: lash_mode_rlm::RlmPromptFeatures {
            images: false,
            ..lash_mode_rlm::RlmPromptFeatures::default()
        },
        ..lash_mode_rlm::RlmModePluginConfig::default()
    }
}

fn build_plugin_stack() -> PluginStack {
    let _clbench_tools = clbench_tool_definitions();
    let mut stack = PluginStack::runtime();
    stack.push(Arc::new(LlmToolsPluginFactory::default()));
    stack.push(Arc::new(StaticPluginFactory::new(
        "clbench_process_handles",
        PluginSpec::new().with_tool_provider(Arc::new(ClbenchProcessHandlesTool)),
    )));
    stack.push(Arc::new(
        SubagentsPluginFactory::new(Arc::new(CapabilityRegistry::new().with(Arc::new(
            StaticCapability::new("explore", SessionSpec::inherit()),
        ))))
        .with_session_spec(SessionSpec::inherit()),
    ));
    stack
}

struct ClbenchProcessHandlesTool;

#[async_trait]
impl ToolProvider for ClbenchProcessHandlesTool {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        vec![list_process_handles_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        (name == "list_process_handles")
            .then(|| Arc::new(list_process_handles_tool_definition().contract()))
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        ToolResult::err_fmt(format_args!(
            "`{}` is handled by the RLM session runtime and cannot run directly",
            call.name
        ))
    }
}

fn clbench_tool_definitions() -> Vec<ToolDefinition> {
    let capabilities = vec!["explore".to_string()];
    vec![
        lash_llm_tools::llm_query_tool_definition(),
        lash_mode_rlm::continue_as_tool_definition(),
        clbench_list_process_handles_tool_definition(),
        lash_subagents::spawn_agent_tool_definition(&capabilities),
    ]
}

fn list_process_handles_tool_definition() -> ToolDefinition {
    clbench_list_process_handles_tool_definition()
}

fn clbench_list_process_handles_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "tool:list_process_handles",
        "list_process_handles",
        "List live lashlang process handles only. Returns `{ monitor: { monitor_id: handle }, tool: { id: handle } }`; terminal, awaited, or cancelled handles are omitted. In CLBench, use this to rediscover live `start call` handles after a handoff or long-running fan-out.",
        ToolDefinition::default_input_schema(),
        json!({
            "type": "object",
            "properties": {
                "monitor": { "type": "object" },
                "tool": { "type": "object" }
            },
            "required": ["monitor", "tool"]
        }),
    )
    .with_examples(vec![r#"handles = (call list_process_handles {})?"#.into()])
    .with_execution_mode(ToolExecutionMode::Parallel)
}

fn build_seed_event(request: &RunnerRequest) -> Option<ModeEvent> {
    let mut globals = serde_json::Map::new();
    if request.init_diary {
        globals.insert("diary".to_string(), Value::Array(Vec::new()));
    }
    (!globals.is_empty()).then(|| {
        lash_mode_rlm::rlm_mode_event(RlmModeEvent::RlmSeed(lash_rlm_types::RlmSeedPluginBody {
            globals,
            projected: Default::default(),
        }))
    })
}

fn build_projected_bindings(
    request: &RunnerRequest,
) -> anyhow::Result<lash_mode_rlm::RlmProjectedBindings> {
    Ok(lash_mode_rlm::RlmProjectedBindings::new()
        .bind_json("iteration", json!(request.iteration))?
        .bind_json("current_query", json!(request.prompt))?
        .bind_json(
            "current_feedback".to_string(),
            request
                .feedback
                .as_ref()
                .map(|feedback| json!(feedback))
                .unwrap_or(Value::Null),
        )?)
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
        TurnOutcome::Stopped(TurnStop::Incomplete) => "incomplete",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn clbench_rlm_tool_surface_exposes_explicit_limited_tools() {
        let core = LashCore::builder()
            .install_mode(ModePreset::rlm_with_config(clbench_rlm_config()))
            .default_mode(ModeId::rlm())
            .model("mock-model", None)
            .max_context_tokens(DEFAULT_MAX_CONTEXT_TOKENS)
            .process_registry(Arc::new(LocalProcessRegistry::default()))
            .plugins(build_plugin_stack())
            .build()
            .expect("core");
        let session = core.session("root").rlm().open().await.expect("session");

        let mut names = session
            .observe()
            .active_tool_definitions()
            .into_iter()
            .map(|tool| tool.name)
            .collect::<Vec<_>>();
        names.sort();
        let mut child_names = clbench_tool_definitions()
            .into_iter()
            .map(|tool| tool.name)
            .collect::<Vec<_>>();
        child_names.sort();
        assert_eq!(
            names,
            vec![
                "continue_as",
                "list_process_handles",
                "llm_query",
                "spawn_agent"
            ]
        );
        assert_eq!(
            child_names, names,
            "CLBench subagents use the same tools as the root agent"
        );

        for denied in [
            "exec_command",
            "read_file",
            "grep",
            "search_web",
            "fetch_url",
            "apply_patch",
            "monitor",
        ] {
            assert!(
                !names.iter().any(|name| name == denied),
                "{denied} must not be callable in CLBench"
            );
        }

        for expected in [
            "llm_query",
            "spawn_agent",
            "continue_as",
            "list_process_handles",
        ] {
            assert!(
                names.iter().any(|name| name == expected),
                "{expected} must be documented in the RLM prompt"
            );
        }
    }
}
