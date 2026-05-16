use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use lash_core::provider::ProviderReliability;
use lash_core::{
    AppendSessionNodesRequest, LashRuntime, LocalBackgroundTaskHost, MessageRole, PluginHost,
    PluginMessage, PluginSpec, ProviderHandle, ProviderOptions, RuntimePersistence,
    SessionAppendNode, SessionPolicy, TurnOutcome,
};
use lash_plugin_observational_memory::ACTIVE_STATE_PLUGIN_TYPE as OM_ACTIVE_STATE_PLUGIN_TYPE;
use lash_provider_openai::OpenAiCompatibleProvider;
use lash_standard_plugins::{
    DefaultPluginStackOptions, DefaultToolSurfaceProfile, default_plugin_stack,
};

use super::openai_compat::OpenAiCompatBenchServer;
use super::providers::{
    BenchmarkEchoTool, BenchmarkLargeToolSurface, benchmark_provider, benchmark_stream_profile,
};
use super::scenarios::RuntimePerfScenario;
use super::store::{RuntimePerfStore, RuntimePerfStoreFactory};

const DEFAULT_PROMPT: &str =
    "Inspect the current state and reply with exactly: runtime perf benchmark ok";
const HISTORY_EXCHANGES: usize = 18;

pub(crate) struct BenchmarkRuntime {
    runtime: LashRuntime,
    _openai_compat_server: Option<OpenAiCompatBenchServer>,
}

impl Deref for BenchmarkRuntime {
    type Target = LashRuntime;

    fn deref(&self) -> &Self::Target {
        &self.runtime
    }
}

impl DerefMut for BenchmarkRuntime {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.runtime
    }
}
pub(crate) fn validate_runtime_perf_turn(
    scenario: RuntimePerfScenario,
    turn_index: usize,
    outcome: &TurnOutcome,
    assistant_output: &lash_core::AssistantOutput,
) -> anyhow::Result<()> {
    let expected = "runtime perf benchmark ok";
    match outcome {
        TurnOutcome::Finished(lash_core::TurnFinish::AssistantMessage { text }) => {
            let valid = if matches!(scenario, RuntimePerfScenario::OpenAiCompatStream) {
                text.contains(expected) || assistant_output.safe_text.contains(expected)
            } else {
                text.trim() == expected || assistant_output.safe_text.trim() == expected
            };
            if valid {
                return Ok(());
            }
            anyhow::bail!(
                "runtime perf scenario {} turn {} produced unexpected assistant text: {:?}",
                scenario.name(),
                turn_index + 1,
                text
            );
        }
        TurnOutcome::Finished(lash_core::TurnFinish::SubmittedValue { value }) => {
            if value.as_str() == Some(expected) {
                return Ok(());
            }
            anyhow::bail!(
                "runtime perf scenario {} turn {} submitted unexpected value: {}",
                scenario.name(),
                turn_index + 1,
                value
            );
        }
        TurnOutcome::Finished(lash_core::TurnFinish::ToolValue { tool_name, value }) => {
            anyhow::bail!(
                "runtime perf scenario {} turn {} finished with tool value from {}: {}",
                scenario.name(),
                turn_index + 1,
                tool_name,
                value
            );
        }
        TurnOutcome::Handoff { session_id } => {
            anyhow::bail!(
                "runtime perf scenario {} turn {} unexpectedly handed off to {}",
                scenario.name(),
                turn_index + 1,
                session_id
            );
        }
        TurnOutcome::Stopped(stop) => {
            anyhow::bail!(
                "runtime perf scenario {} turn {} stopped with {:?}; assistant_output={:?}",
                scenario.name(),
                turn_index + 1,
                stop,
                assistant_output
            );
        }
    }
}

pub(crate) fn build_embed_core(
    scenario: RuntimePerfScenario,
    store: Arc<RuntimePerfStore>,
) -> anyhow::Result<lash::LashCore> {
    let mut builder = match scenario {
        RuntimePerfScenario::EmbedStandard => lash::LashCore::standard(),
        RuntimePerfScenario::EmbedRlm => lash::LashCore::rlm()
            .tools(Arc::new(BenchmarkEchoTool))
            .default_mode(lash::ModeId::rlm()),
        _ => anyhow::bail!("{} is not an embed scenario", scenario.name()),
    };
    builder = builder
        .provider(benchmark_provider(scenario).into_handle())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .store_factory(Arc::new(RuntimePerfStoreFactory { store }));
    builder.build().map_err(anyhow::Error::from)
}

pub(crate) async fn build_runtime(
    scenario: RuntimePerfScenario,
) -> anyhow::Result<BenchmarkRuntime> {
    let execution_mode = scenario.execution_mode();
    let standard_context_approach = scenario.standard_context_approach();
    let openai_compat_server = if matches!(scenario, RuntimePerfScenario::OpenAiCompatStream) {
        Some(OpenAiCompatBenchServer::start(benchmark_stream_profile(scenario)).await?)
    } else {
        None
    };
    let base_url = openai_compat_server
        .as_ref()
        .map(|server| server.base_url.clone())
        .unwrap_or_else(|| "https://example.invalid/v1".to_string());
    let provider: ProviderHandle = match scenario {
        RuntimePerfScenario::OpenAiCompatStream => ProviderHandle::new(
            OpenAiCompatibleProvider::new("test-key", base_url.clone())
                .with_options(ProviderOptions {
                    reliability: ProviderReliability::disabled(),
                    ..ProviderOptions::default()
                })
                .into_components(),
        ),
        _ => benchmark_provider(scenario).into_handle(),
    };
    let policy = SessionPolicy {
        model: "mock-model".to_string(),
        provider,
        max_context_tokens: Some(200_000),
        execution_mode: execution_mode.clone(),
        standard_context_approach: standard_context_approach.clone(),
        ..SessionPolicy::default()
    };

    let profile =
        DefaultToolSurfaceProfile::for_runtime(standard_context_approach.as_ref(), false, false);
    let store = Arc::new(RuntimePerfStore::default()) as Arc<dyn RuntimePersistence>;
    let factories = default_plugin_stack(DefaultPluginStackOptions {
        execution_mode,
        standard_context_approach: standard_context_approach.clone(),
        bundles: profile.bundles,
        tavily_api_key: None,
    });
    let mut factories = factories.into_factories();
    factories.push(Arc::new(lash_core::BuiltinTaskControlsPluginFactory::new()));
    factories.push(Arc::new(lash_core::BuiltinMonitorToolPluginFactory::new()));
    factories.push(Arc::new(lash_core::plugin::StaticPluginFactory::new(
        "runtime_perf_tools",
        PluginSpec::new().with_tool_provider(Arc::new(BenchmarkEchoTool)),
    )));
    if matches!(scenario, RuntimePerfScenario::RlmLargeToolSurface) {
        factories.push(Arc::new(lash_core::plugin::StaticPluginFactory::new(
            "runtime_perf_large_tool_surface",
            PluginSpec::new().with_tool_provider(Arc::new(BenchmarkLargeToolSurface)),
        )));
    }
    factories.push(Arc::new(
        lash_mode_standard::BuiltinStandardModePluginFactory,
    ));
    factories.push(Arc::new(
        lash_mode_rlm::BuiltinRlmModePluginFactory::default(),
    ));
    let plugin_host = PluginHost::new(factories);
    let builder = LashRuntime::builder()
        .with_policy(policy.clone())
        .with_store(Arc::clone(&store))
        .with_background_task_host(Arc::new(LocalBackgroundTaskHost::default()))
        .with_plugin_host(plugin_host);
    let runtime = builder.build().await?;
    Ok(BenchmarkRuntime {
        runtime,
        _openai_compat_server: openai_compat_server,
    })
}

pub(crate) async fn seed_runtime_state(
    runtime: &mut LashRuntime,
    scenario: RuntimePerfScenario,
) -> anyhow::Result<()> {
    let mut nodes = Vec::with_capacity(HISTORY_EXCHANGES * 2);
    for index in 0..HISTORY_EXCHANGES {
        nodes.push(SessionAppendNode::message(PluginMessage::text(
            MessageRole::User,
            format!(
                "Historical user turn {index}: trace the performance-sensitive path through runtime/session graph/tool prep."
            ),
        )));
        nodes.push(SessionAppendNode::message(PluginMessage::text(
            MessageRole::Assistant,
            format!(
                "Historical assistant turn {index}: inspected runtime.rs, turn_runner.rs, and token ledger export surfaces."
            ),
        )));
    }

    runtime
        .append_session_nodes(AppendSessionNodesRequest {
            nodes,
            requires_ancestor_node_id: None,
        })
        .await
        .map_err(|err| anyhow::anyhow!("seed historical messages: {err}"))?;

    if matches!(scenario, RuntimePerfScenario::ObservationalMemory) {
        let observed_through_message_id = runtime
            .read_view()
            .messages()
            .last()
            .map(|message| message.id.clone())
            .ok_or_else(|| anyhow::anyhow!("OM scenario expected seeded messages"))?;
        runtime
            .append_session_nodes(AppendSessionNodesRequest {
                nodes: vec![SessionAppendNode::plugin(
                    OM_ACTIVE_STATE_PLUGIN_TYPE,
                    serde_json::json!({
                        "observed_through_message_id": observed_through_message_id,
                        "observations": "Date: Apr 13, 2026\n* 🔴 User is evaluating Lash runtime performance without model inference.\n* 🟡 Historical inspection focused on runtime, token ledger export, and tool initialization overhead.\n* ✅ Shared process-wide search cache was added for indexed grep state.",
                        "current_task": "Primary: Benchmark runtime overhead.\nSecondary: Compare standard, rlm, and observational memory paths.",
                        "suggested_response": "Report the runtime benchmark numbers and the dominant overheads directly."
                    }),
                )],
                requires_ancestor_node_id: None,
            })
            .await
            .map_err(|err| anyhow::anyhow!("seed OM reflection node: {err}"))?;
    }

    Ok(())
}

pub(crate) async fn prepare_turn(
    runtime: &mut LashRuntime,
    scenario: RuntimePerfScenario,
    turn_index: usize,
) -> anyhow::Result<()> {
    if !matches!(scenario, RuntimePerfScenario::RlmGlobals) {
        return Ok(());
    }

    let _ = runtime;
    let _ = turn_index;
    Ok(())
}

pub(crate) fn rlm_perf_projected_bindings(
    scenario: RuntimePerfScenario,
    turn_index: usize,
) -> anyhow::Result<lash_mode_rlm::RlmProjectedBindings> {
    Ok(lash_mode_rlm::RlmProjectedBindings::new()
        .bind_json(
            "benchmark",
            serde_json::json!({
                "name": "runtime_perf",
                "scenario": scenario.name(),
            }),
        )?
        .bind_json(
            "input",
            serde_json::json!({
                "turn": turn_index + 1,
                "goal": "measure runtime overhead across a longer same-session chat",
                "path": "crates/lash/src/runtime",
            }),
        )?
        .bind_json(
            "chat",
            serde_json::json!({
                "turn_count": turn_index + 1,
                "scenario": scenario.name(),
                "mode": "runtime_perf",
            }),
        )?)
}

pub(crate) fn benchmark_prompt(scenario: RuntimePerfScenario, turn_index: usize) -> String {
    match scenario {
        RuntimePerfScenario::Standard | RuntimePerfScenario::EmbedStandard => format!(
            "Turn {} of a longer runtime benchmark conversation. Inspect the state and reply with exactly: {}",
            turn_index + 1,
            DEFAULT_PROMPT
                .rsplit_once(": ")
                .map(|(_, text)| text)
                .unwrap_or("runtime perf benchmark ok")
        ),
        RuntimePerfScenario::Rlm | RuntimePerfScenario::EmbedRlm => format!(
            "Turn {} in RLM mode. Continue the benchmark chat and reply with exactly: {}",
            turn_index + 1,
            DEFAULT_PROMPT
                .rsplit_once(": ")
                .map(|(_, text)| text)
                .unwrap_or("runtime perf benchmark ok")
        ),
        RuntimePerfScenario::RlmLargeToolSurface => format!(
            "Turn {} in RLM mode with a Gmail-sized callable tool surface. Do not call a Gmail tool; reply with exactly: {}",
            turn_index + 1,
            DEFAULT_PROMPT
                .rsplit_once(": ")
                .map(|(_, text)| text)
                .unwrap_or("runtime perf benchmark ok")
        ),
        RuntimePerfScenario::RlmToolCalls => format!(
            "Turn {} in RLM mode. Exercise the benchmark_echo tool path and reply with exactly: {}",
            turn_index + 1,
            DEFAULT_PROMPT
                .rsplit_once(": ")
                .map(|(_, text)| text)
                .unwrap_or("runtime perf benchmark ok")
        ),
        RuntimePerfScenario::RlmGlobals => format!(
            "Turn {} in RLM mode with bound variables updated for this turn. Inspect the current state and reply with exactly: {}",
            turn_index + 1,
            DEFAULT_PROMPT
                .rsplit_once(": ")
                .map(|(_, text)| text)
                .unwrap_or("runtime perf benchmark ok")
        ),
        RuntimePerfScenario::ObservationalMemory => format!(
            "Turn {} in observational memory mode. Continue the same longer benchmark conversation and reply with exactly: {}",
            turn_index + 1,
            DEFAULT_PROMPT
                .rsplit_once(": ")
                .map(|(_, text)| text)
                .unwrap_or("runtime perf benchmark ok")
        ),
        RuntimePerfScenario::OpenAiCompatStream => format!(
            "Turn {} in OpenAI-compatible streaming benchmark mode. Continue the benchmark chat and reply with exactly: runtime perf benchmark ok",
            turn_index + 1
        ),
        RuntimePerfScenario::TurnCheckpoint => format!(
            "Turn {} in turn durability checkpoint benchmark mode. Checkpoint and restore pending effects, then reply with exactly: runtime perf benchmark ok",
            turn_index + 1
        ),
    }
}
