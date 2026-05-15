use std::sync::Arc;

use crate::config::LashConfig;
use lash::advanced::ExecutionMode;
use lash::plugins::{
    BuiltinMonitorToolPluginFactory, BuiltinTaskControlsPluginFactory, PluginFactory,
};
use lash::prompt::{
    PromptBuiltin, PromptContribution, PromptSlot, PromptTemplate, PromptTemplateEntry,
    PromptTemplateSection,
};
use lash::provider::ProviderHandle;
#[cfg(test)]
use lash::tools::{
    ToolAvailabilityConfig, ToolCall, ToolContract, ToolDefinition, ToolExecutionMode,
    ToolManifest, ToolProvider, ToolResult,
};
use lash::tracing::TraceLevel;
use lash::{PluginStack, SessionSpec};
use lash_core::{
    FileAttachmentStore, LocalBackgroundTaskHost, PromptLayer, SessionPolicy, ToolState,
};
use lash_llm_tools::LlmToolsPluginFactory;
use lash_plugin_mcp::McpPluginFactory;
use lash_plugin_plan_mode::{PlanModePluginFactory, UpdatePlanPluginFactory};
use lash_plugin_prompt_context::{PromptContextPluginConfig, PromptContextPluginFactory};
use lash_plugin_ui_activity::UiActivityPluginFactory;
use lash_provider_openai::{OpenAiCompatibleProvider, OpenAiProvider};
use lash_standard_plugins::{
    DefaultPluginStackOptions, DefaultToolSurfaceProfile, default_plugin_stack,
};
use lash_subagents::{LocalSubagentHost, SubagentHost, SubagentsPluginFactory, default_registry};
use lash_tui::Terminal;
use serde_json::Value as JsonValue;

use crate::autonomous::{AutonomousPersistenceContext, run_autonomous};
use crate::instructions::{FsInstructionSource, InstructionLoaderConfig};
use crate::interactive::run_app;
use crate::prompt_tool::{CliPromptBridge, cli_ask_plugin_factory};
use crate::session_bootstrap::{CliSessionOpener, SessionBootstrapSource};
use crate::{Args, SkillCatalog, setup};
use crate::{
    apply_standard_context_approach_overrides, cleanup_terminal, ensure_supported_execution_mode,
    hash12, info_text, info_text_unconfigured, models_dev_catalog, parse_execution_mode,
    parse_model_selection, parse_standard_context_approach, resolve_model_selection,
    resolve_model_variant, validate_model_selection,
};
use lash_plugin_prompt_context::InstructionSource;

const CLI_AUTONOMOUS_INTRO: &str = "You are an autonomous AI coding assistant running without a human in the loop.\nComplete the task end-to-end without asking for user input.\nIf the task is incomplete and concrete next actions are available, continue executing them instead of stopping to summarize incompletion.";
const CLI_AUTONOMOUS_EXECUTION: &str = "- No user is available during this run. Default to acting without asking. Ask only when progress is blocked and user intervention is strictly required; otherwise make the best reasonable decision from local context and continue.\n- Do not stop merely to report that work remains. If concrete actions are still available, keep executing them.\n- Only summarize remaining work when you are blocked, need a decision, or have exhausted feasible actions for this turn.\n- Do not claim completion unless you have actually verified the required end state.";
const CLI_RLM_SUBMISSION_GUIDANCE: &str = "- When calling `submit`, keep the submitted value concise. Do not include large variables such as diffs, full logs, raw command output, or other bulky dumps unless the user explicitly asks for them. Prefer short prose. If you use `format`, use it with small values rather than large captured variables.";

fn openai_shortcut_provider(api_key: String, base_url: &str) -> ProviderHandle {
    if base_url.trim().is_empty() {
        ProviderHandle::new(OpenAiProvider::new(api_key).into_components())
    } else {
        ProviderHandle::new(OpenAiCompatibleProvider::new(api_key, base_url).into_components())
    }
}

struct PluginFactorySurfaceInput<'a> {
    autonomous: bool,
    execution_mode: ExecutionMode,
    tavily_key: String,
    instruction_source: Arc<dyn InstructionSource>,
    session_policy: SessionPolicy,
    lash_config: &'a LashConfig,
    host_docs_dir: Option<std::path::PathBuf>,
    prompt_bridge: CliPromptBridge,
}

struct PluginFactoriesForSurface {
    stack: PluginStack,
    subagent_host: Arc<dyn SubagentHost>,
}

fn plugin_factories_for_surface(input: PluginFactorySurfaceInput<'_>) -> PluginFactoriesForSurface {
    let PluginFactorySurfaceInput {
        autonomous,
        execution_mode,
        tavily_key,
        instruction_source,
        session_policy,
        lash_config,
        host_docs_dir,
        prompt_bridge,
    } = input;
    let subagent_host: Arc<dyn SubagentHost> = Arc::new(LocalSubagentHost::default());

    let capability_registry = Arc::new(default_registry(&lash_config.agent_models));

    let profile = DefaultToolSurfaceProfile::for_runtime(
        session_policy.standard_context_approach.as_ref(),
        !autonomous,
        !tavily_key.is_empty(),
    );
    let mut plugin_stack = default_plugin_stack(DefaultPluginStackOptions {
        execution_mode,
        standard_context_approach: session_policy.standard_context_approach.clone(),
        bundles: profile.bundles.clone(),
        tavily_api_key: if tavily_key.is_empty() {
            None
        } else {
            Some(tavily_key)
        },
    });
    plugin_stack.push(Arc::new(PromptContextPluginFactory::new(
        Arc::clone(&instruction_source),
        PromptContextPluginConfig::default(),
    )) as Arc<dyn PluginFactory>);
    if profile.interactive_extras {
        if let Some(host_docs_dir) = host_docs_dir {
            plugin_stack.push(
                Arc::new(crate::host_docs::HostDocsPluginFactory::new(host_docs_dir))
                    as Arc<dyn PluginFactory>,
            );
        }
        plugin_stack.push(Arc::new(
            PlanModePluginFactory::new(Default::default())
                .with_prompt(Arc::new(prompt_bridge.clone())),
        ));
        plugin_stack.push(cli_ask_plugin_factory(prompt_bridge));
        plugin_stack.push(Arc::new(UiActivityPluginFactory));
        // `update_plan` drives the sticky plan dock at the bottom of
        // the TUI. Interactive-only here; root-only inside the plugin
        // itself (the factory returns an inert plugin for subagent
        // / compaction / other non-root sessions).
        plugin_stack.push(Arc::new(UpdatePlanPluginFactory));
    }
    plugin_stack.push(Arc::new(lash_autoresearch::AutoresearchPluginFactory));
    plugin_stack.push(Arc::new(BuiltinTaskControlsPluginFactory::new()));
    plugin_stack.push(Arc::new(BuiltinMonitorToolPluginFactory::new()));
    plugin_stack.push(Arc::new(LlmToolsPluginFactory::default()));
    plugin_stack.push(Arc::new(
        SubagentsPluginFactory::new(capability_registry, Arc::clone(&subagent_host))
            .with_session_spec(SessionSpec::inherit()),
    ));
    PluginFactoriesForSurface {
        stack: plugin_stack,
        subagent_host,
    }
}

fn autonomous_tool_allowed(name: &str) -> bool {
    !matches!(name, "ask" | "showcase")
        && !name.starts_with("plan_")
        && name != "request_user_input"
}

async fn apply_autonomous_tool_policy(session: &lash::LashSession) -> anyhow::Result<()> {
    let mut snapshot = session.control().tools().state().await?;
    retain_autonomous_tools(&mut snapshot);
    session
        .control()
        .tools()
        .advanced()
        .apply_state(snapshot)
        .await?;
    Ok(())
}

fn retain_autonomous_tools(snapshot: &mut ToolState) {
    snapshot.retain(|name, _| autonomous_tool_allowed(name));
}

fn parse_rlm_var_arg(raw: &str) -> Result<(String, JsonValue), String> {
    let Some((name, json)) = raw.split_once('=') else {
        return Err(format!(
            "Invalid `--rlm-var` value `{raw}`. Expected `NAME=JSON`."
        ));
    };
    let name = name.trim();
    if name.is_empty() {
        return Err("`--rlm-var` requires a non-empty variable name.".to_string());
    }
    let value = serde_json::from_str::<JsonValue>(json.trim())
        .map_err(|err| format!("Invalid JSON for `--rlm-var {name}=...`: {err}"))?;
    Ok((name.to_string(), value))
}

fn resolve_rlm_projected_bindings(
    args: &Args,
) -> Result<Option<lash_mode_rlm::RlmProjectedBindings>, String> {
    let mut bindings = lash_mode_rlm::RlmProjectedBindings::new();
    let mut has_bindings = false;
    if let Some(path) = &args.rlm_vars_file {
        let raw = std::fs::read_to_string(path)
            .map_err(|err| format!("Could not read `--rlm-vars-file {}`: {err}", path.display()))?;
        let value = serde_json::from_str::<JsonValue>(&raw).map_err(|err| {
            format!(
                "Invalid JSON in `--rlm-vars-file {}`: {err}",
                path.display()
            )
        })?;
        let object = value.as_object().ok_or_else(|| {
            format!(
                "`--rlm-vars-file {}` must contain a top-level JSON object.",
                path.display()
            )
        })?;
        for (name, value) in object {
            bindings = bindings
                .bind_json(name.clone(), value.clone())
                .map_err(|err| err.to_string())?;
            has_bindings = true;
        }
    }
    for raw in &args.rlm_var {
        let (name, value) = parse_rlm_var_arg(raw)?;
        bindings = bindings
            .bind_json(name, value)
            .map_err(|err| err.to_string())?;
        has_bindings = true;
    }

    if !has_bindings {
        return Ok(None);
    }
    Ok(Some(bindings))
}

fn cli_prompt_config(autonomous: bool, execution_mode: &ExecutionMode) -> PromptLayer {
    let mut intro_entries = vec![PromptTemplateEntry::builtin(PromptBuiltin::MainAgentIntro)];
    if autonomous {
        intro_entries.push(PromptTemplateEntry::slot(PromptSlot::CliAutonomousIntro));
    }
    intro_entries.push(PromptTemplateEntry::slot(PromptSlot::Intro));

    let mut execution_entries = vec![
        PromptTemplateEntry::builtin(PromptBuiltin::ExecutionInstructions),
        PromptTemplateEntry::slot(PromptSlot::Execution),
    ];
    if autonomous {
        execution_entries.push(PromptTemplateEntry::slot(
            PromptSlot::CliAutonomousExecution,
        ));
    }
    if *execution_mode == ExecutionMode::new("rlm") {
        execution_entries.push(PromptTemplateEntry::slot(PromptSlot::CliRlmExecution));
    }

    let template = PromptTemplate::new(vec![
        PromptTemplateSection::untitled(intro_entries),
        PromptTemplateSection::titled("Execution", execution_entries),
        PromptTemplateSection::titled(
            "Guidance",
            vec![
                PromptTemplateEntry::builtin(PromptBuiltin::CoreGuidance),
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
    ]);

    let mut layer = PromptLayer::with_template(template);
    if autonomous {
        layer.add_contribution(PromptContribution::new(
            PromptSlot::CliAutonomousIntro,
            "",
            CLI_AUTONOMOUS_INTRO,
        ));
        layer.add_contribution(PromptContribution::new(
            PromptSlot::CliAutonomousExecution,
            "",
            CLI_AUTONOMOUS_EXECUTION,
        ));
    }
    if *execution_mode == ExecutionMode::new("rlm") {
        layer.add_contribution(PromptContribution::new(
            PromptSlot::CliRlmExecution,
            "RLM Submit Output",
            CLI_RLM_SUBMISSION_GUIDANCE,
        ));
    }
    layer
}

pub(crate) async fn run(args: Args) -> anyhow::Result<()> {
    lash_providers_builtin::register_all();
    // Handle --reset before any TUI/provider setup
    if args.reset {
        use std::io::Write;

        // Design system ANSI colors
        const SODIUM: &str = "\x1b[38;2;232;163;60m"; // #e8a33c
        const CHALK: &str = "\x1b[38;2;232;228;208m"; // #e8e4d0
        const ASH_TEXT: &str = "\x1b[38;2;90;90;80m"; // #5a5a50
        const LICHEN: &str = "\x1b[38;2;138;158;108m"; // #8a9e6c
        const ERR: &str = "\x1b[38;2;204;68;68m"; // #c44
        const BOLD: &str = "\x1b[1m";
        const RESET: &str = "\x1b[0m";

        let lash_dir = crate::paths::lash_home();
        let cache_dir = crate::paths::lash_cache_dir();

        eprintln!();
        eprintln!("  {SODIUM}{BOLD}/ reset{RESET}");
        eprintln!();
        eprintln!("  {ERR}This will permanently delete all lash data:{RESET}");
        eprintln!();
        eprintln!(
            "    {ASH_TEXT}config, credentials   {CHALK}{}{RESET}",
            lash_dir.display()
        );
        eprintln!(
            "    {ASH_TEXT}runtime cache          {CHALK}{}{RESET}",
            cache_dir.display()
        );
        eprintln!();
        eprint!("  {SODIUM}Are you sure? [y/N]{RESET} ");
        std::io::stderr().flush()?;

        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer)?;
        if answer.trim().eq_ignore_ascii_case("y") {
            if lash_dir.exists() {
                std::fs::remove_dir_all(&lash_dir)?;
            }
            if cache_dir.exists() {
                std::fs::remove_dir_all(&cache_dir)?;
            }
            eprintln!("  {LICHEN}Done.{RESET} All data removed.");
        } else {
            eprintln!("  {ASH_TEXT}Aborted.{RESET}");
        }
        eprintln!();
        return Ok(());
    }

    if args.check_update {
        println!("{}", crate::update::check_update_text().await?);
        return Ok(());
    }

    if args.update {
        crate::update::install_latest_release().await?;
        return Ok(());
    }

    // Resolve config before TUI init (may need interactive terminal)
    let existing_config = LashConfig::load(&crate::paths::config_file());
    let shortcut_api_key = args.api_key.clone().or_else(|| {
        if args.base_url.trim().is_empty() {
            std::env::var("OPENAI_API_KEY").ok()
        } else {
            std::env::var("OPENAI_COMPATIBLE_API_KEY")
                .ok()
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        }
    });
    if args.info && existing_config.is_none() && shortcut_api_key.is_none() {
        let execution_mode =
            ensure_supported_execution_mode(match args.execution_mode.as_deref() {
                Some(raw) => parse_execution_mode(raw).map_err(anyhow::Error::msg)?,
                None => ExecutionMode::standard(),
            })
            .map_err(anyhow::Error::msg)?;
        let cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| ".".to_string());
        println!("{}", info_text_unconfigured(&execution_mode, &cwd));
        return Ok(());
    }
    let interactive_startup = !args.info && args.print_prompt.is_none();
    let startup_system_message: Option<String> = None;
    let (mut lash_config, mut active_provider) = if args.provider || existing_config.is_none() {
        if let Some(ref key) = shortcut_api_key {
            let provider = openai_shortcut_provider(key.clone(), &args.base_url);
            let mut cfg = existing_config
                .clone()
                .unwrap_or_else(|| LashConfig::new(&provider));
            cfg.upsert_provider(&provider);
            let _ = cfg.set_active_provider_kind(provider.kind());
            cfg.set_tavily_api_key(args.tavily_api_key.clone());
            (cfg, provider)
        } else {
            let cfg = setup::run_setup_with_existing(existing_config.as_ref()).await?;
            let provider = cfg
                .build_active_provider()
                .map_err(|err| anyhow::anyhow!("build active provider: {err}"))?;
            (cfg, provider)
        }
    } else {
        // SAFETY: else branch means existing_config.is_some() (checked above)
        #[allow(clippy::unnecessary_unwrap)]
        let c = existing_config.unwrap();
        let provider = c
            .build_active_provider()
            .map_err(|err| anyhow::anyhow!("build active provider: {err}"))?;
        (c, provider)
    };

    // CLI env/flags override stored config
    if let Some(ref key) = args.tavily_api_key {
        lash_config.set_tavily_api_key(Some(key.clone()));
    }
    if args.print_prompt.is_none() {
        crate::expose_provider_thinking(&mut active_provider);
        lash_config.save(&crate::paths::config_file())?;
    }
    let host_docs = if interactive_startup {
        Some(
            crate::host_docs::ensure_host_docs()
                .map_err(|err| anyhow::anyhow!("failed to prepare Lash CLI host docs: {err}"))?,
        )
    } else {
        None
    };
    let model_catalog = models_dev_catalog().map_err(anyhow::Error::msg)?;
    if let Err(err) = model_catalog
        .refresh_if_stale(lash_core::model_info::DEFAULT_REFRESH_INTERVAL)
        .await
    {
        eprintln!("warning: failed to refresh models.dev catalog: {err}");
    }

    let configured_model_default = lash_config
        .model_default(active_provider.kind())
        .filter(|selection| !selection.model.trim().is_empty());
    let requested_model = args
        .model
        .clone()
        .or_else(|| configured_model_default.map(|selection| selection.model.clone()))
        .unwrap_or_else(|| active_provider.default_model().to_string());
    let selection = parse_model_selection(&requested_model).map_err(anyhow::Error::msg)?;
    validate_model_selection(&active_provider, &selection).map_err(anyhow::Error::msg)?;
    let resolved_model_spec =
        resolve_model_selection(&active_provider, &selection, model_catalog.as_ref())
            .map_err(anyhow::Error::msg)?;
    let model = selection.model.clone();
    let requested_variant = args.variant.as_deref().or_else(|| {
        if args.model.is_none() {
            configured_model_default
                .filter(|default| default.model == model)
                .and_then(|default| default.variant.as_deref())
        } else {
            None
        }
    });
    let model_variant = resolve_model_variant(&active_provider, &model, requested_variant)
        .map_err(anyhow::Error::msg)?;
    if args.resume.is_none()
        && args.print_prompt.is_none()
        && (args.model.is_some() || args.variant.is_some())
    {
        lash_config.set_model_default(active_provider.kind(), model.clone(), model_variant.clone());
        lash_config.save(&crate::paths::config_file())?;
    }
    let trace_level = TraceLevel::from(args.trace_level);
    let trace_path =
        if crate::detailed_debug_logging_enabled(args.debug) || trace_level.is_extended() {
            let dir = crate::paths::lash_home().join("sessions");
            Some(dir.join(format!(
                "{}.trace.jsonl",
                chrono::Local::now().format("%Y%m%d_%H%M%S")
            )))
        } else {
            None
        };

    let autonomous = args.print_prompt.is_some();
    if !autonomous && (args.await_background_work || args.turn_usage_json.is_some()) {
        return Err(anyhow::anyhow!(
            "`--await-background-work` and `--turn-usage-json` require `--print`."
        ));
    }
    let session_bootstrap_probe = if args.info {
        None
    } else {
        Some(crate::session_bootstrap::SessionBootstrap::open(
            SessionBootstrapSource::from_resume_arg(args.resume.clone()),
        )?)
    };
    if let Some(session_bootstrap_probe) = session_bootstrap_probe.as_ref() {
        tracing::debug!(
            session_file = session_bootstrap_probe.filename(),
            resumed = args.resume.is_some(),
            trace_path = ?trace_path,
            debug_logging = crate::detailed_debug_logging_enabled(args.debug),
            "prepared session bootstrap"
        );
    }
    // Autonomous runs still need a stable session id so provider-side prompt caching
    // and benchmark accounting can key repeated requests within the same session.
    let run_session_id = session_bootstrap_probe
        .as_ref()
        .and_then(|bootstrap| bootstrap.run_session_id());
    let persisted_session_config = session_bootstrap_probe
        .as_ref()
        .and_then(|bootstrap| bootstrap.persisted_config());

    // Execution mode: CLI flag wins, then persisted config, then Standard.
    // Resolved BEFORE building the plugin host + runtime so the lashlang
    // thread starts correctly and plugins see the right mode on resume.
    let execution_mode = match args.execution_mode.as_deref() {
        Some(raw) => {
            ensure_supported_execution_mode(parse_execution_mode(raw).map_err(anyhow::Error::msg)?)
                .map_err(anyhow::Error::msg)?
        }
        None => persisted_session_config
            .as_ref()
            .map(|config| config.execution_mode.clone())
            .and_then(|mode| crate::ensure_supported_execution_mode(mode).ok())
            .unwrap_or(ExecutionMode::standard()),
    };
    let has_om_overrides = args.om_observation_message_tokens.is_some()
        || args.om_observation_buffer_tokens.is_some()
        || args.om_observation_block_after_tokens.is_some()
        || args.om_observation_max_tokens_per_batch.is_some()
        || args.om_previous_observer_tokens.is_some()
        || args.om_reflection_observation_tokens.is_some()
        || args.om_reflection_buffer_activation_percent.is_some()
        || args.om_reflection_block_after_tokens.is_some();
    if execution_mode != ExecutionMode::standard()
        && (args.standard_context_approach.is_some() || has_om_overrides)
    {
        return Err(anyhow::anyhow!(
            "`--context-approach` and OM tuning flags only apply to `--execution-mode standard`."
        ));
    }
    let configured_standard_context_approach = if execution_mode == ExecutionMode::standard() {
        let approach = match args.standard_context_approach.as_deref() {
            Some(raw) => parse_standard_context_approach(raw).map_err(anyhow::Error::msg)?,
            None => persisted_session_config
                .as_ref()
                .and_then(|config| config.standard_context_approach.clone())
                .unwrap_or_default(),
        };
        Some(
            apply_standard_context_approach_overrides(
                approach,
                args.om_observation_message_tokens,
                args.om_observation_buffer_tokens,
                args.om_observation_block_after_tokens,
                args.om_observation_max_tokens_per_batch,
                args.om_previous_observer_tokens,
                args.om_reflection_observation_tokens,
                args.om_reflection_buffer_activation_percent,
                args.om_reflection_block_after_tokens,
            )
            .map_err(anyhow::Error::msg)?,
        )
    } else {
        None
    };
    let rlm_projected_bindings =
        resolve_rlm_projected_bindings(&args).map_err(anyhow::Error::msg)?;
    let rlm_globals_supported = execution_mode == ExecutionMode::new("rlm");
    if rlm_projected_bindings.is_some() && !rlm_globals_supported {
        return Err(anyhow::anyhow!(
            "`--rlm-var` and `--rlm-vars-file` require `--execution-mode rlm`."
        ));
    }
    let instruction_source: Arc<dyn InstructionSource> =
        Arc::new(FsInstructionSource::with_config(InstructionLoaderConfig {
            global_root: Some(crate::paths::lash_home()),
            ..Default::default()
        }));
    let prompt_layer = cli_prompt_config(autonomous, &execution_mode);
    let session_policy = SessionPolicy {
        model: model.clone(),
        provider: active_provider.clone(),
        model_variant,
        max_context_tokens: Some(resolved_model_spec.context_window() as usize),
        session_id: run_session_id.clone(),
        autonomous,
        execution_mode: execution_mode.clone(),
        standard_context_approach: configured_standard_context_approach.clone(),
        ..Default::default()
    };
    let tavily_key = lash_config.tavily_api_key().unwrap_or_default().to_string();
    let prompt_bridge = CliPromptBridge::default();

    let PluginFactoriesForSurface {
        stack: mut plugin_stack,
        subagent_host,
    } = plugin_factories_for_surface(PluginFactorySurfaceInput {
        autonomous,
        execution_mode: execution_mode.clone(),
        tavily_key,
        instruction_source: Arc::clone(&instruction_source),
        session_policy: session_policy.clone(),
        lash_config: &lash_config,
        host_docs_dir: host_docs.as_ref().map(|docs| docs.dir().to_path_buf()),
        prompt_bridge: prompt_bridge.clone(),
    });
    let mcp_factory = Arc::new(
        McpPluginFactory::new(lash_config.mcp_servers().clone())
            .await
            .map_err(|err| anyhow::anyhow!("failed to connect MCP servers: {err}"))?,
    );
    plugin_stack.push(mcp_factory);
    if args.info {
        let cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| ".".to_string());
        println!(
            "{}",
            info_text(
                &active_provider,
                &model,
                session_policy.model_variant.as_deref(),
                &execution_mode,
                session_policy.standard_context_approach.as_ref(),
                Some(resolved_model_spec.context_window()),
                None,
                &cwd,
                None,
                None,
                None,
            )
        );
        return Ok(());
    }
    let runtime_factory = CliSessionOpener::new(
        plugin_stack.clone(),
        prompt_layer,
        Arc::new(FileAttachmentStore::new(crate::paths::attachments_dir())),
        trace_path,
        trace_level,
        Arc::new(LocalBackgroundTaskHost::default()),
    );
    let opened_session = runtime_factory
        .open(
            SessionBootstrapSource::from_resume_arg(args.resume.clone()),
            session_policy.clone(),
        )
        .await?;
    let session = opened_session.session;
    if autonomous {
        apply_autonomous_tool_policy(&session)
            .await
            .map_err(|err| anyhow::anyhow!("failed to apply autonomous tool policy: {err}"))?;
    }
    let active_tool_definitions = session.control().tools().active_definitions().await?;
    let toolset_hash =
        hash12(&serde_json::to_vec(&active_tool_definitions).unwrap_or_else(|_| b"[]".to_vec()));
    let initial_policy = session.policy_snapshot();
    let initial_model_variant = initial_policy.model_variant.clone();
    let store = opened_session.bootstrap.store();
    let session_name = opened_session.bootstrap.session_name();
    let mut logger = opened_session.logger;
    session.control().tools().refresh_surface().await?;
    if rlm_projected_bindings.is_some() && args.print_prompt.is_none() {
        return Err(anyhow::anyhow!(
            "`--rlm-var` and `--rlm-vars-file` are currently supported for autonomous `--print-prompt` turns; interactive hosts should use the RLM projection API."
        ));
    }

    // ── Autonomous preset: skip TUI, run session, print response to stdout ──
    if let Some(prompt) = args.print_prompt {
        return run_autonomous(
            session,
            prompt,
            SkillCatalog::from_dirs(&crate::paths::default_skill_dirs()),
            AutonomousPersistenceContext {
                await_background_work: args.await_background_work,
                turn_usage_json: args.turn_usage_json.clone(),
            },
            rlm_projected_bindings,
        )
        .await;
    }

    // Install panic hook that restores the terminal
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        cleanup_terminal();
        default_hook(info);
    }));

    // Initialize terminal
    let terminal = Terminal::enter()?;

    run_app(
        terminal,
        session,
        runtime_factory,
        lash_config.clone(),
        prompt_bridge,
        &mut logger,
        &args,
        active_provider.clone(),
        model,
        resolved_model_spec.context_window(),
        session_name,
        model_catalog,
        Arc::clone(&store),
        toolset_hash,
        initial_model_variant,
        execution_mode,
        startup_system_message,
        subagent_host,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::ToolRegistry;

    struct DummyToolProvider;

    fn dummy_tool(name: &str) -> ToolDefinition {
        ToolDefinition::raw(
            name,
            format!("{name} description"),
            ToolDefinition::default_input_schema(),
            serde_json::json!({ "type": "null" }),
        )
        .with_availability(ToolAvailabilityConfig::callable())
        .with_execution_mode(ToolExecutionMode::Parallel)
    }

    #[async_trait::async_trait]
    impl ToolProvider for DummyToolProvider {
        fn tool_manifests(&self) -> Vec<ToolManifest> {
            dummy_tools()
                .into_iter()
                .map(|tool| tool.manifest())
                .collect()
        }

        fn resolve_contract(&self, name: &str) -> Option<std::sync::Arc<ToolContract>> {
            dummy_tools()
                .into_iter()
                .find(|tool| tool.name == name)
                .map(|tool| std::sync::Arc::new(tool.contract()))
        }

        async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
            ToolResult::err_fmt(format_args!("unexpected tool call: {}", call.name))
        }
    }

    fn dummy_tools() -> Vec<ToolDefinition> {
        vec![
            dummy_tool("read_file"),
            dummy_tool("ask"),
            dummy_tool("plan_exit"),
            dummy_tool("showcase"),
        ]
    }

    fn plugin_factory_ids_for_autonomous(autonomous: bool) -> Vec<&'static str> {
        let provider =
            ProviderHandle::new(OpenAiCompatibleProvider::new("test", "").into_components());
        let lash_config = LashConfig::new(&provider);
        plugin_factories_for_surface(PluginFactorySurfaceInput {
            autonomous,
            execution_mode: ExecutionMode::standard(),
            tavily_key: String::new(),
            instruction_source: Arc::new(FsInstructionSource::default()),
            session_policy: SessionPolicy {
                provider,
                ..Default::default()
            },
            lash_config: &lash_config,
            host_docs_dir: (!autonomous)
                .then(|| std::path::PathBuf::from("/tmp/lash-home/docs/lash-cli")),
            prompt_bridge: CliPromptBridge::default(),
        })
        .stack
        .into_factories()
        .into_iter()
        .map(|factory| factory.id())
        .collect()
    }

    #[test]
    fn cli_surface_stack_does_not_install_mode_factories() {
        let ids = plugin_factory_ids_for_autonomous(false);

        assert!(!ids.contains(&"mode_standard"));
        assert!(!ids.contains(&"mode_rlm"));
    }

    #[test]
    fn host_docs_prompt_is_interactive_only() {
        let interactive_ids = plugin_factory_ids_for_autonomous(false);
        assert!(interactive_ids.contains(&"lash_cli_host_docs"));

        let autonomous_ids = plugin_factory_ids_for_autonomous(true);
        assert!(!autonomous_ids.contains(&"lash_cli_host_docs"));
    }

    #[test]
    fn rlm_prompt_config_uses_cli_rlm_slot_and_contribution() {
        let layer = cli_prompt_config(false, &ExecutionMode::new("rlm"));
        let template = layer.template.as_ref().expect("cli prompt template");
        let contributions = layer
            .slots
            .values()
            .flat_map(|slot| slot.contributions.iter())
            .collect::<Vec<_>>();

        let execution = template
            .sections
            .iter()
            .find(|section| section.title.as_deref() == Some("Execution"))
            .expect("execution section");
        assert!(execution.entries.iter().any(|entry| {
            matches!(
                entry,
                PromptTemplateEntry::Slot {
                    slot: PromptSlot::CliRlmExecution
                }
            )
        }));
        assert!(contributions.iter().any(|contribution| {
            contribution.slot == PromptSlot::CliRlmExecution
                && contribution.content.as_ref() == CLI_RLM_SUBMISSION_GUIDANCE
        }));
    }

    #[test]
    fn standard_prompt_config_omits_cli_rlm_slot_and_contribution() {
        let layer = cli_prompt_config(false, &ExecutionMode::standard());
        let template = layer.template.as_ref().expect("cli prompt template");
        let contributions = layer
            .slots
            .values()
            .flat_map(|slot| slot.contributions.iter())
            .collect::<Vec<_>>();

        assert!(!template.sections.iter().any(|section| {
            section.entries.iter().any(|entry| {
                matches!(
                    entry,
                    PromptTemplateEntry::Slot {
                        slot: PromptSlot::CliRlmExecution
                    }
                )
            })
        }));
        assert!(
            !contributions
                .iter()
                .any(|contribution| contribution.slot == PromptSlot::CliRlmExecution)
        );
    }

    #[test]
    fn autonomous_prompt_config_uses_autonomous_slots() {
        let layer = cli_prompt_config(true, &ExecutionMode::standard());
        let template = layer.template.as_ref().expect("cli prompt template");
        let contributions = layer
            .slots
            .values()
            .flat_map(|slot| slot.contributions.iter())
            .collect::<Vec<_>>();

        assert!(template.sections.iter().any(|section| {
            section.entries.iter().any(|entry| {
                matches!(
                    entry,
                    PromptTemplateEntry::Slot {
                        slot: PromptSlot::CliAutonomousIntro
                    }
                )
            })
        }));
        assert!(template.sections.iter().any(|section| {
            section.entries.iter().any(|entry| {
                matches!(
                    entry,
                    PromptTemplateEntry::Slot {
                        slot: PromptSlot::CliAutonomousExecution
                    }
                )
            })
        }));
        assert!(contributions.iter().any(|contribution| {
            contribution.slot == PromptSlot::CliAutonomousIntro
                && contribution.content.as_ref() == CLI_AUTONOMOUS_INTRO
        }));
        assert!(contributions.iter().any(|contribution| {
            contribution.slot == PromptSlot::CliAutonomousExecution
                && contribution.content.as_ref() == CLI_AUTONOMOUS_EXECUTION
        }));
    }

    #[test]
    fn api_key_shortcut_selects_direct_openai_without_base_url() {
        let provider = openai_shortcut_provider("key".to_string(), "");
        assert_eq!(provider.kind(), "openai");
        assert_eq!(provider.to_spec().kind, "openai");
        assert!(provider.to_spec().config.get("base_url").is_none());
    }

    #[test]
    fn api_key_shortcut_selects_compatible_provider_with_base_url() {
        let provider = openai_shortcut_provider("key".to_string(), "https://example.invalid/v1");
        let spec = provider.to_spec();
        assert_eq!(provider.kind(), "openai-compatible");
        assert_eq!(spec.kind, "openai-compatible");
        assert_eq!(spec.config["base_url"], "https://example.invalid/v1");
    }

    #[test]
    fn autonomous_policy_disables_interactive_tools() {
        let tool_registry = ToolRegistry::from_tool_provider(Arc::new(DummyToolProvider)).unwrap();

        let mut snapshot = tool_registry.export_state();
        retain_autonomous_tools(&mut snapshot);
        assert!(snapshot.contains("read_file"));
        assert!(!snapshot.contains("ask"));
        assert!(!snapshot.contains("plan_exit"));
        assert!(!snapshot.contains("showcase"));
    }
}
