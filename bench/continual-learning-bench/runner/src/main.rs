use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use clap::Parser;
use lash::plugin::{PluginFactory, PluginSpec, StaticPluginFactory};
use lash::provider::LashConfig;
use lash::{
    AppendSessionNodesRequest, BackgroundRuntimeHost, BuiltinToolResultProjectionPluginFactory,
    EmbeddedRuntimeHost, EventSink, ExecutionMode, FollowedTurn, InputItem, LashRuntime,
    NoopEventSink, PersistedSessionState, PersistentRuntimeServices, PluginHost, ProviderHandle,
    RuntimeCoreConfig, RuntimePersistence, SessionAppendNode, SessionEventRecord, SessionPolicy,
    SessionUsageReport, StandardContextApproach, TokenLedgerEntry, TokioSessionTaskExecutor,
    ToolDefinition, ToolExecutionMode, ToolProvider, ToolResult, TurnFinish, TurnInjectionBridge,
    TurnInput, TurnInputInjectionBridge, TurnOutcome,
};
use lash_harness_opt::clbench::CLBENCH_MEMORY_GUIDANCE;
use lash_llm_tools::LlmToolsPluginFactory;
use lash_mode_rlm::RlmTurnInputExt;
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
                .with_prompt_template(lash_harness_opt::clbench::clbench_prompt_template(
                    CLBENCH_MEMORY_GUIDANCE,
                )),
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

    if let Some(defaults) = build_globals_patch(&request) {
        runtime
            .append_session_nodes(AppendSessionNodesRequest {
                nodes: vec![SessionAppendNode::event(SessionEventRecord::Mode(
                    lash::ModeEvent::rlm(RlmModeEvent::RlmGlobalsPatch(defaults)),
                ))],
                requires_ancestor_node_id: None,
            })
            .await
            .context("bind clbench defaults")?;
    }

    let sink = NoopEventSink;
    let followed = runtime
        .stream_turn_following_handoffs(
            (TurnInput {
                items: vec![InputItem::Text {
                    text: clbench_turn_text(&request),
                }],
                image_blobs: Default::default(),
                mode: None,
                mode_turn_options: Some(lash::ModeTurnOptions::typed(
                    ExecutionMode::new("rlm"),
                    RlmTermination::SubmitRequired {
                        schema: Some(request.response_schema.clone()),
                    },
                )?),
                trace_turn_id: Some(format!("clbench-turn-{:04}", request.iteration)),
                mode_extension: None,
                turn_context: lash::TurnContext::default(),
            })
            .rlm_project(build_projected_bindings(&request)?)?,
            &sink as &dyn EventSink,
            tokio_util::sync::CancellationToken::new(),
        )
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

fn usage_from_followed_turn(followed: &FollowedTurn) -> SessionUsageReport {
    let entries = followed
        .turns
        .iter()
        .filter(|turn| turn.token_usage.total() != 0 || turn.token_usage.cached_input_tokens != 0)
        .map(|turn| TokenLedgerEntry {
            source: "turn".to_string(),
            model: turn.state.policy.model.clone(),
            usage: turn.token_usage.clone(),
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

fn build_plugin_session(
    execution_mode: ExecutionMode,
    policy: &SessionPolicy,
) -> Result<Arc<lash::PluginSession>> {
    let clbench_tools = clbench_tool_definitions();
    let factories: Vec<Arc<dyn PluginFactory>> = vec![
        Arc::new(BuiltinToolResultProjectionPluginFactory::default()),
        Arc::new(lash_mode_rlm::BuiltinRlmModePluginFactory::new(
            lash_mode_rlm::RlmModePluginConfig {
                prompt_features: lash_mode_rlm::RlmPromptFeatures {
                    images: false,
                    ..lash_mode_rlm::RlmPromptFeatures::default()
                },
                ..lash_mode_rlm::RlmModePluginConfig::default()
            },
        )),
        Arc::new(LlmToolsPluginFactory),
        Arc::new(StaticPluginFactory::new(
            "clbench_async_handles",
            PluginSpec::new().with_tool_provider(Arc::new(ClbenchAsyncHandlesTool)),
        )),
        Arc::new(SubagentsPluginFactory::new(
            policy.clone(),
            Arc::new(
                CapabilityRegistry::new().with(Arc::new(StaticCapability::new(
                    "explore",
                    CapabilitySpec {
                        model: CapabilityField::Inherit,
                        model_variant: CapabilityOptionalField::Inherit,
                        execution_mode: CapabilityField::Inherit,
                        tool_surface: CapabilityToolSurface::Explicit(clbench_tools.clone()),
                        recursion: CapabilityRecursion::Inherit,
                    },
                ))),
            ),
            Arc::new(LocalSubagentHost::default()) as Arc<dyn SubagentHost>,
        )),
    ];
    PluginHost::new(factories)
        .with_background_tasks()
        .build_session(
            "root",
            execution_mode,
            None::<StandardContextApproach>,
            None,
        )
        .context("build plugin session")
}

struct ClbenchAsyncHandlesTool;

#[async_trait]
impl ToolProvider for ClbenchAsyncHandlesTool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![list_async_handles_tool_definition()]
    }

    async fn execute(&self, name: &str, _args: &Value) -> ToolResult {
        ToolResult::err_fmt(format_args!(
            "`{name}` is handled by the RLM session runtime and cannot run directly"
        ))
    }
}

fn clbench_tool_definitions() -> Vec<ToolDefinition> {
    let capabilities = vec!["explore".to_string()];
    vec![
        lash_llm_tools::llm_query_tool_definition(),
        lash_mode_rlm::continue_as_tool_definition(),
        clbench_list_async_handles_tool_definition(),
        lash_subagents::spawn_agent_tool_definition(&capabilities),
    ]
}

fn list_async_handles_tool_definition() -> ToolDefinition {
    clbench_list_async_handles_tool_definition()
}

fn clbench_list_async_handles_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "list_async_handles",
        "List live lashlang async handles only. Returns `{ monitor: { monitor_id: handle }, subagent: { name: handle }, tool: { id: handle } }`; terminal, awaited, or cancelled handles are omitted. In CLBench, use this to rediscover live `start call` handles after a handoff or long-running fan-out.",
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

fn build_globals_patch(request: &RunnerRequest) -> Option<RlmGlobalsPatchPluginBody> {
    let mut set_default = serde_json::Map::new();
    if request.init_diary {
        set_default.insert("diary".to_string(), Value::Array(Vec::new()));
    }
    (!set_default.is_empty()).then_some(RlmGlobalsPatchPluginBody { set_default })
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
        TurnOutcome::Stopped(lash::TurnStop::Cancelled) => "interrupted",
        TurnOutcome::Stopped(_) => "failed",
    }
}

fn done_reason_label(outcome: &TurnOutcome) -> &'static str {
    match outcome {
        TurnOutcome::Finished(TurnFinish::AssistantMessage { .. }) => "assistant_message",
        TurnOutcome::Finished(TurnFinish::SubmittedValue { .. }) => "submitted_value",
        TurnOutcome::Finished(TurnFinish::ToolValue { .. }) => "tool_value",
        TurnOutcome::Handoff { .. } => "handoff",
        TurnOutcome::Stopped(lash::TurnStop::Cancelled) => "cancelled",
        TurnOutcome::Stopped(lash::TurnStop::InvalidInput) => "invalid_input",
        TurnOutcome::Stopped(lash::TurnStop::MaxTurns) => "max_turns",
        TurnOutcome::Stopped(lash::TurnStop::ToolFailure) => "tool_failure",
        TurnOutcome::Stopped(lash::TurnStop::ProviderError) => "provider_error",
        TurnOutcome::Stopped(lash::TurnStop::PluginAbort) => "plugin_abort",
        TurnOutcome::Stopped(lash::TurnStop::RuntimeError) => "runtime_error",
        TurnOutcome::Stopped(lash::TurnStop::SubmittedError { .. }) => "submitted_error",
        TurnOutcome::Stopped(lash::TurnStop::ToolError { .. }) => "tool_error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clbench_rlm_tool_surface_exposes_explicit_limited_tools() {
        let mode = ExecutionMode::new("rlm");
        let policy = SessionPolicy {
            execution_mode: mode.clone(),
            ..SessionPolicy::default()
        };
        let session = build_plugin_session(mode.clone(), &policy).expect("plugin session");
        let surface = session.tool_surface("root", mode);

        let mut names = surface.tool_names();
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
                "list_async_handles",
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
                !surface.has_callable_tool(denied),
                "{denied} must not be callable in CLBench"
            );
        }

        let docs = surface.prompt_tool_docs();
        for expected in [
            "llm_query",
            "spawn_agent",
            "continue_as",
            "list_async_handles",
        ] {
            assert!(
                docs.contains(expected),
                "{expected} must be documented in the RLM prompt"
            );
        }
        assert!(
            !docs.contains("peer"),
            "CLBench prompt must not advertise unavailable peer capability"
        );

        let spawn_agent = surface
            .callable_tools_iter()
            .find(|tool| tool.name == "spawn_agent")
            .expect("spawn_agent tool");
        assert_eq!(
            spawn_agent.input_schema["properties"]["capability"]["enum"],
            json!(["explore"])
        );
    }
}
