use std::sync::Arc;

use lash::{
    LashCore, ModeId, ModePreset,
    advanced::{PluginMessage, StandardContextApproach, TurnOutcome},
    messages::MessageRole,
    persistence::SessionStateEnvelope,
    plugins::{
        BuiltinMonitorToolPluginFactory, BuiltinTaskControlsPluginFactory, PluginSpec,
        StaticPluginFactory,
    },
    provider::{ProviderHandle, ProviderOptions, ProviderReliability},
};
use lash_llm_tools::LlmToolsPluginFactory;
use lash_plugin_observational_memory::ACTIVE_STATE_PLUGIN_TYPE as OM_ACTIVE_STATE_PLUGIN_TYPE;
use lash_provider_openai::OpenAiCompatibleProvider;
use lash_standard_plugins::{StandardToolStackOptions, standard_tool_stack};

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
    core: LashCore,
    session: Option<lash::LashSession>,
    store: Arc<RuntimePerfStore>,
    _openai_compat_server: Option<OpenAiCompatBenchServer>,
}

impl BenchmarkRuntime {
    pub(crate) fn usage_report(&self) -> lash::usage::SessionUsageReport {
        self.session
            .as_ref()
            .expect("benchmark session")
            .usage_report()
    }

    pub(crate) fn read_view(&self) -> lash::persistence::SessionReadView {
        self.session
            .as_ref()
            .expect("benchmark session")
            .read_view()
    }

    pub(crate) fn store(&self) -> Arc<RuntimePerfStore> {
        Arc::clone(&self.store)
    }

    pub(crate) fn core(&self) -> LashCore {
        self.core.clone()
    }

    pub(crate) async fn reopen_with_state(
        &mut self,
        scenario: RuntimePerfScenario,
        state: lash::persistence::PersistedSessionState,
    ) -> anyhow::Result<()> {
        if let Some(session) = self.session.take() {
            session.close().await?;
        }
        let store = Arc::clone(&self.store) as Arc<dyn lash::persistence::RuntimePersistence>;
        self.session = Some(
            self.core
                .session(format!("runtime-perf-{}", scenario.name()))
                .mode(ModeId::new(scenario.execution_mode().plugin_id()))
                .store(store)
                .open_with_state(state)
                .await?,
        );
        Ok(())
    }

    pub(crate) async fn set_turn_phase_probe(
        &self,
        probe: Arc<dyn lash::advanced::RuntimeTurnPhaseProbe>,
    ) {
        self.session
            .as_ref()
            .expect("benchmark session")
            .set_turn_phase_probe(probe)
            .await;
    }

    pub(crate) async fn run_turn(
        &self,
        input: lash::TurnInput,
        cancel: tokio_util::sync::CancellationToken,
    ) -> anyhow::Result<lash::TurnResult> {
        self.session
            .as_ref()
            .expect("benchmark session")
            .turn(input)
            .cancel(cancel)
            .collect_session_events_with(&lash::advanced::NoopEventSink)
            .await
            .map_err(anyhow::Error::from)
    }

    pub(crate) async fn run_turn_with_effect_scope(
        &self,
        input: lash::TurnInput,
        cancel: tokio_util::sync::CancellationToken,
        effect_scope: lash::advanced::RuntimeEffectControllerScope<'_>,
    ) -> anyhow::Result<lash::TurnResult> {
        self.session
            .as_ref()
            .expect("benchmark session")
            .turn(input)
            .cancel(cancel)
            .run_with_effect_scope(effect_scope)
            .await
            .map(|output| output.result)
            .map_err(anyhow::Error::from)
    }

    pub(crate) async fn await_background_work(&self) -> anyhow::Result<()> {
        self.session
            .as_ref()
            .expect("benchmark session")
            .background_tasks()
            .await_all()
            .await?;
        Ok(())
    }

    pub(crate) async fn export_state(&self) -> SessionStateEnvelope {
        self.session
            .as_ref()
            .expect("benchmark session")
            .control()
            .state()
            .export()
            .await
    }
}
pub(crate) fn validate_runtime_perf_turn(
    scenario: RuntimePerfScenario,
    turn_index: usize,
    outcome: &TurnOutcome,
    assistant_output: &lash::turn::AssistantOutput,
) -> anyhow::Result<()> {
    let expected = "runtime perf benchmark ok";
    match outcome {
        TurnOutcome::Finished(lash::advanced::TurnFinish::AssistantMessage { text }) => {
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
        TurnOutcome::Finished(lash::advanced::TurnFinish::SubmittedValue { value }) => {
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
        TurnOutcome::Finished(lash::advanced::TurnFinish::ToolValue { tool_name, value }) => {
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
            .tools(Arc::new(BenchmarkEchoTool::default()))
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
    build_runtime_with_store(scenario, None).await
}

pub(crate) async fn build_runtime_with_store(
    scenario: RuntimePerfScenario,
    store: Option<Arc<RuntimePerfStore>>,
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
    let mode_id = ModeId::new(execution_mode.plugin_id());
    let store = store.unwrap_or_else(|| Arc::new(RuntimePerfStore::default()));
    let mut plugin_stack = standard_tool_stack(StandardToolStackOptions {
        standard_context_approach: standard_context_approach.clone(),
        tavily_api_key: None,
    });
    plugin_stack.push(Arc::new(BuiltinTaskControlsPluginFactory::new()));
    plugin_stack.push(Arc::new(BuiltinMonitorToolPluginFactory::new()));
    plugin_stack.push(Arc::new(StaticPluginFactory::new(
        "runtime_perf_tools",
        PluginSpec::new().with_tool_provider(Arc::new(BenchmarkEchoTool::default())),
    )));
    if matches!(scenario, RuntimePerfScenario::RlmLlmQuery) {
        plugin_stack.push(Arc::new(LlmToolsPluginFactory::default()));
    }
    if matches!(scenario, RuntimePerfScenario::RlmLargeToolSurface) {
        plugin_stack.push(Arc::new(StaticPluginFactory::new(
            "runtime_perf_large_tool_surface",
            PluginSpec::new().with_tool_provider(Arc::new(BenchmarkLargeToolSurface::default())),
        )));
    }
    let mut builder = LashCore::builder()
        .install_mode(mode_preset(&mode_id, standard_context_approach)?)
        .default_mode(mode_id.clone())
        .provider(provider)
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .plugins(plugin_stack);
    // RlmGlobals profiles live per-turn projected bindings. Store-backed turns
    // reject live mode extensions because they cannot be checkpointed/resumed.
    if !matches!(scenario, RuntimePerfScenario::RlmGlobals) {
        builder = builder.store_factory(Arc::new(RuntimePerfStoreFactory {
            store: Arc::clone(&store),
        }));
    }
    let core = if matches!(scenario, RuntimePerfScenario::StoreReopen) {
        builder
            .advanced()
            .residency(lash::advanced::Residency::ActivePathOnly)
            .build()?
    } else {
        builder.build()?
    };
    let session = core
        .session(format!("runtime-perf-{}", scenario.name()))
        .mode(mode_id)
        .open()
        .await?;
    Ok(BenchmarkRuntime {
        core,
        session: Some(session),
        store,
        _openai_compat_server: openai_compat_server,
    })
}

fn mode_preset(
    mode: &ModeId,
    standard_context_approach: Option<StandardContextApproach>,
) -> anyhow::Result<ModePreset> {
    match mode.as_str() {
        "standard" => {
            Ok(ModePreset::standard().with_standard_context_approach(standard_context_approach))
        }
        "rlm" => Ok(ModePreset::rlm()),
        other => anyhow::bail!("unsupported runtime perf mode `{other}`"),
    }
}

pub(crate) async fn seed_runtime_state(
    runtime: &mut BenchmarkRuntime,
    scenario: RuntimePerfScenario,
) -> anyhow::Result<()> {
    let mut messages = Vec::with_capacity(HISTORY_EXCHANGES * 2);
    for index in 0..HISTORY_EXCHANGES {
        messages.push(PluginMessage::text(
            MessageRole::User,
            format!(
                "Historical user turn {index}: trace the performance-sensitive path through runtime/session graph/tool prep."
            ),
        ));
        messages.push(PluginMessage::text(
            MessageRole::Assistant,
            format!(
                "Historical assistant turn {index}: inspected runtime.rs, turn_runner.rs, and token ledger export surfaces."
            ),
        ));
    }

    runtime
        .session
        .as_ref()
        .expect("benchmark session")
        .control()
        .state()
        .append_messages(messages)
        .await
        .map_err(|err| anyhow::anyhow!("seed historical messages: {err}"))?;

    if matches!(
        scenario,
        RuntimePerfScenario::ObservationalMemory
            | RuntimePerfScenario::ObservationalMemoryMaintenance
    ) {
        if matches!(
            scenario,
            RuntimePerfScenario::ObservationalMemoryMaintenance
        ) {
            return Ok(());
        }

        let observed_through_message_id = runtime
            .read_view()
            .messages()
            .last()
            .map(|message| message.id.clone())
            .ok_or_else(|| anyhow::anyhow!("OM scenario expected seeded messages"))?;
        runtime
            .session
            .as_ref()
            .expect("benchmark session")
            .control()
            .state()
            .append_plugin_body(
                OM_ACTIVE_STATE_PLUGIN_TYPE,
                serde_json::json!({
                    "observed_through_message_id": observed_through_message_id,
                    "observations": "Date: Apr 13, 2026\n* 🔴 User is evaluating Lash runtime performance without model inference.\n* 🟡 Historical inspection focused on runtime, token ledger export, and tool initialization overhead.\n* ✅ Shared process-wide search cache was added for indexed grep state.",
                    "current_task": "Primary: Benchmark runtime overhead.\nSecondary: Compare standard, rlm, and observational memory paths.",
                    "suggested_response": "Report the runtime benchmark numbers and the dominant overheads directly."
                }),
            )
            .await
            .map_err(|err| anyhow::anyhow!("seed OM reflection node: {err}"))?;
    }

    Ok(())
}

pub(crate) async fn prepare_turn(
    runtime: &mut BenchmarkRuntime,
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
        RuntimePerfScenario::StandardToolCalls => format!(
            "Turn {} in standard mode. Use the batch tool to exercise parallel benchmark_echo calls, then reply with exactly: {}",
            turn_index + 1,
            DEFAULT_PROMPT
                .rsplit_once(": ")
                .map(|(_, text)| text)
                .unwrap_or("runtime perf benchmark ok")
        ),
        RuntimePerfScenario::RlmAsyncHandles => format!(
            "Turn {} in RLM mode. Exercise start/await/cancel async handles, then submit exactly: {}",
            turn_index + 1,
            DEFAULT_PROMPT
                .rsplit_once(": ")
                .map(|(_, text)| text)
                .unwrap_or("runtime perf benchmark ok")
        ),
        RuntimePerfScenario::RlmLlmQuery => format!(
            "Turn {} in RLM mode. Exercise llm_query direct completion, then submit exactly: {}",
            turn_index + 1,
            DEFAULT_PROMPT
                .rsplit_once(": ")
                .map(|(_, text)| text)
                .unwrap_or("runtime perf benchmark ok")
        ),
        RuntimePerfScenario::RlmToolRetry => format!(
            "Turn {} in RLM mode. Exercise benchmark_retry and submit exactly: {}",
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
        RuntimePerfScenario::ObservationalMemoryMaintenance => format!(
            "Turn {} in observational memory maintenance benchmark mode. Leave the hidden observer maintenance path eligible after persistence and reply with exactly: runtime perf benchmark ok",
            turn_index + 1
        ),
        RuntimePerfScenario::OpenAiCompatStream => format!(
            "Turn {} in OpenAI-compatible streaming benchmark mode. Continue the benchmark chat and reply with exactly: runtime perf benchmark ok",
            turn_index + 1
        ),
        RuntimePerfScenario::TurnCheckpoint => format!(
            "Turn {} in turn durability checkpoint benchmark mode. Checkpoint and restore pending effects, then reply with exactly: runtime perf benchmark ok",
            turn_index + 1
        ),
        RuntimePerfScenario::ScopedEffectController => format!(
            "Turn {} in scoped effect-controller benchmark mode. Continue the benchmark chat and reply with exactly: runtime perf benchmark ok",
            turn_index + 1
        ),
        RuntimePerfScenario::StoreReopen => format!(
            "Turn {} in store reopen benchmark mode. Continue after persisted reload and reply with exactly: runtime perf benchmark ok",
            turn_index + 1
        ),
    }
}
