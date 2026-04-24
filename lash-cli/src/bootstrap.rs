use std::sync::Arc;

use lash::provider::{LashConfig, ProviderHandle};
use lash::*;
use lash_default_tools::{
    DefaultToolPluginOptions, DefaultToolSurfaceProfile, tool_plugin_factories,
};
use lash_plugin_plan_mode::{PlanModePluginFactory, UpdatePlanPluginFactory};
use lash_plugin_prompt_context::{PromptContextPluginConfig, PromptContextPluginFactory};
use lash_plugin_ui_activity::UiActivityPluginFactory;
use lash_provider_openai::OpenAiGenericProvider;
use lash_subagents::{LocalSubagentHost, SubagentHost, SubagentsPluginFactory, default_registry};
use lash_tui::Terminal;
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::autonomous::{AutonomousPersistenceContext, run_autonomous};
use crate::interactive::run_app;
use crate::session_bootstrap::{SessionBootstrap, SessionBootstrapSource};
use crate::session_log::DbSessionStoreFactory;
use crate::{Args, setup};
use crate::{
    apply_context_approach_overrides, autonomous_prompt_template, cleanup_terminal,
    ensure_supported_execution_mode, hash12, info_text, info_text_unconfigured, models_dev_catalog,
    parse_context_approach, parse_execution_mode, parse_model_selection, provider_display_label,
    resolve_model_selection, resolve_model_variant, validate_model_selection,
};

fn plugin_factories_for_surface(
    autonomous: bool,
    execution_mode: ExecutionMode,
    tavily_key: String,
    instruction_source: Arc<dyn InstructionSource>,
    session_policy: SessionPolicy,
    lash_config: &LashConfig,
    host_docs_dir: std::path::PathBuf,
) -> Vec<Arc<dyn PluginFactory>> {
    let low_tier_execution_mode = lash_config
        .runtime
        .low_tier_subagent_execution_mode
        .unwrap_or(ExecutionMode::Standard);
    let capability_registry = Arc::new(default_registry(
        &lash_config.agent_models,
        low_tier_execution_mode,
    ));
    let subagent_host: Arc<dyn SubagentHost> = Arc::new(LocalSubagentHost::default());

    let profile = DefaultToolSurfaceProfile::for_runtime(
        &session_policy.context_approach,
        !autonomous,
        !tavily_key.is_empty(),
    );
    let mut plugin_factories = tool_plugin_factories(DefaultToolPluginOptions {
        execution_mode,
        context_approach: session_policy.context_approach.clone(),
        bundles: profile.bundles.clone(),
        tavily_api_key: if tavily_key.is_empty() {
            None
        } else {
            Some(tavily_key)
        },
        instruction_source: Some(Arc::clone(&instruction_source)),
    });
    plugin_factories.push(Arc::new(PromptContextPluginFactory::new(
        Arc::clone(&instruction_source),
        PromptContextPluginConfig::default(),
    )) as Arc<dyn PluginFactory>);
    plugin_factories.push(
        Arc::new(crate::host_docs::HostDocsPluginFactory::new(host_docs_dir))
            as Arc<dyn PluginFactory>,
    );
    if profile.interactive_extras {
        plugin_factories.push(Arc::new(PlanModePluginFactory::new(Default::default())));
        plugin_factories.push(Arc::new(UiActivityPluginFactory));
        // `update_plan` drives the sticky plan dock at the bottom of
        // the TUI. Interactive-only here; root-only inside the plugin
        // itself (the factory returns an inert plugin for subagent
        // / compaction / other non-root sessions).
        plugin_factories.push(Arc::new(UpdatePlanPluginFactory));
    }
    plugin_factories.push(Arc::new(lash_autoresearch::AutoresearchPluginFactory));
    plugin_factories.push(Arc::new(
        lash_mode_standard::BuiltinStandardModePluginFactory,
    ));
    plugin_factories.push(Arc::new(
        lash_mode_rlm::BuiltinRlmModePluginFactory::default(),
    ));
    plugin_factories.push(Arc::new(SubagentsPluginFactory::new(
        session_policy,
        capability_registry,
        subagent_host,
    )));
    plugin_factories
}

fn autonomous_tool_allowed(name: &str) -> bool {
    !matches!(name, "ask" | "show_snippet_to_user" | "showcase")
        && !name.starts_with("plan_")
        && name != "request_user_input"
}

fn apply_autonomous_tool_policy(
    dynamic_tools: &DynamicToolProvider,
) -> Result<(), ReconfigureError> {
    let mut snapshot = dynamic_tools.export_state();
    snapshot
        .tools
        .retain(|name, _| autonomous_tool_allowed(name));
    dynamic_tools.apply_state(snapshot).map(|_| ())
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

fn resolve_rlm_globals_patch(
    args: &Args,
) -> Result<Option<lash::RlmGlobalsPatchPluginBody>, String> {
    let mut set = JsonMap::new();
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
        set.extend(object.clone());
    }
    for raw in &args.rlm_var {
        let (name, value) = parse_rlm_var_arg(raw)?;
        set.insert(name, value);
    }

    let unset = args
        .rlm_unset
        .iter()
        .map(|name| name.trim())
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    let patch = lash::RlmGlobalsPatchPluginBody { set, unset };
    if patch.is_empty() {
        return Ok(None);
    }
    Ok(Some(patch))
}

pub(crate) async fn run(args: Args, prompt_template: PromptTemplate) -> anyhow::Result<()> {
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
    if args.info && existing_config.is_none() && args.api_key.is_none() {
        let execution_mode =
            ensure_supported_execution_mode(match args.execution_mode.as_deref() {
                Some(raw) => parse_execution_mode(raw).map_err(anyhow::Error::msg)?,
                None => ExecutionMode::Standard,
            })
            .map_err(anyhow::Error::msg)?;
        let cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| ".".to_string());
        println!("{}", info_text_unconfigured(execution_mode, &cwd));
        return Ok(());
    }
    let interactive_startup = !args.info && args.print_prompt.is_none();
    let mut startup_system_message: Option<String> = None;
    let (mut lash_config, active_provider) = if args.provider || existing_config.is_none() {
        if let Some(ref key) = args.api_key {
            // Shortcut: env var or --api-key activates OpenAI-compatible directly.
            let provider = ProviderHandle::new(Box::new(OpenAiGenericProvider::new(
                key.clone(),
                args.base_url.clone(),
            )));
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
        let mut provider = c
            .build_active_provider()
            .map_err(|err| anyhow::anyhow!("build active provider: {err}"))?;
        match provider.ensure_fresh().await {
            Ok(true) => {
                let mut c = c;
                c.upsert_provider(&provider);
                c.save(&crate::paths::config_file())?;
                (c, provider)
            }
            Ok(false) => (c, provider),
            Err(err) => {
                if interactive_startup {
                    let provider_label = provider_display_label(&provider);
                    startup_system_message = Some(format!(
                        "{} refresh failed: {}. Lash opened in recovery mode. Use /provider or relaunch with --provider to reauthenticate or switch providers.",
                        provider_label, err
                    ));
                    (c, provider)
                } else {
                    return Err(err.into());
                }
            }
        }
    };

    // CLI env/flags override stored config
    if let Some(ref key) = args.tavily_api_key {
        lash_config.set_tavily_api_key(Some(key.clone()));
    }
    if args.print_prompt.is_none() {
        lash_config.save(&crate::paths::config_file())?;
    }
    let host_docs = crate::host_docs::ensure_host_docs()
        .map_err(|err| anyhow::anyhow!("failed to prepare Lash CLI host docs: {err}"))?;
    let model_catalog = models_dev_catalog().map_err(anyhow::Error::msg)?;
    if let Err(err) = model_catalog
        .refresh_if_stale(lash::model_info::DEFAULT_REFRESH_INTERVAL)
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
    let llm_log_path = if crate::detailed_debug_logging_enabled(args.debug) {
        let dir = crate::paths::lash_home().join("sessions");
        Some(dir.join(format!(
            "{}.llm.jsonl",
            chrono::Local::now().format("%Y%m%d_%H%M%S")
        )))
    } else {
        None
    };

    let session_bootstrap =
        SessionBootstrap::open(SessionBootstrapSource::from_resume_arg(args.resume.clone()))?;
    tracing::debug!(
        session_file = session_bootstrap.filename(),
        resumed = args.resume.is_some(),
        llm_log_path = ?llm_log_path,
        debug_logging = crate::detailed_debug_logging_enabled(args.debug),
        "prepared session bootstrap"
    );
    let autonomous = args.print_prompt.is_some();
    if !autonomous && (args.await_background_work || args.turn_usage_json.is_some()) {
        return Err(anyhow::anyhow!(
            "`--await-background-work` and `--turn-usage-json` require `--print`."
        ));
    }
    let prompt_template = if autonomous {
        autonomous_prompt_template()
    } else {
        prompt_template
    };
    let store = session_bootstrap.store();
    // Autonomous runs still need a stable session id so provider-side prompt caching
    // and benchmark accounting can key repeated requests within the same session.
    let run_session_id = session_bootstrap.run_session_id();

    // Peek the persisted session config so we can use the correct
    // execution mode and context approach BEFORE building the plugin
    // host and runtime. Without this, resuming a Rlm session from a
    // Standard-mode bootstrap would fail to start the lashlang thread
    // and would build plugins with the wrong mode.
    let persisted_session_config = session_bootstrap.persisted_config();

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
            .map(|config| config.execution_mode)
            .and_then(|mode| crate::ensure_supported_execution_mode(mode).ok())
            .unwrap_or(ExecutionMode::Standard),
    };

    let configured_context_approach = match args.context_approach.as_deref() {
        Some(raw) => parse_context_approach(raw).map_err(anyhow::Error::msg)?,
        None => persisted_session_config
            .as_ref()
            .map(|config| config.context_approach.clone())
            .unwrap_or_default(),
    };
    let configured_context_approach = apply_context_approach_overrides(
        configured_context_approach,
        args.om_observation_message_tokens,
        args.om_observation_buffer_tokens,
        args.om_observation_block_after_tokens,
        args.om_observation_max_tokens_per_batch,
        args.om_previous_observer_tokens,
        args.om_reflection_observation_tokens,
        args.om_reflection_buffer_activation_percent,
        args.om_reflection_block_after_tokens,
    )
    .map_err(anyhow::Error::msg)?;
    let rlm_globals_patch = resolve_rlm_globals_patch(&args).map_err(anyhow::Error::msg)?;
    if rlm_globals_patch.is_some() && !matches!(execution_mode, ExecutionMode::Rlm) {
        return Err(anyhow::anyhow!(
            "`--rlm-var`, `--rlm-vars-file`, and `--rlm-unset` require `--execution-mode rlm`."
        ));
    }

    let instruction_source: Arc<dyn InstructionSource> =
        Arc::new(FsInstructionSource::with_config(InstructionLoaderConfig {
            global_root: Some(crate::paths::lash_home()),
            ..Default::default()
        }));
    let session_policy = SessionPolicy {
        model: model.clone(),
        provider: active_provider.clone(),
        model_variant,
        max_context_tokens: Some(resolved_model_spec.context_window() as usize),
        session_id: run_session_id.clone(),
        execution_mode,
        context_approach: configured_context_approach.clone(),
        ..Default::default()
    };
    let host_core = RuntimeCoreConfig::default()
        .with_prompt_template(prompt_template)
        .with_llm_log_path(llm_log_path)
        .with_credential_store_path(Some(crate::paths::config_file()));

    let tavily_key = lash_config.tavily_api_key().unwrap_or_default().to_string();
    let turn_injection_bridge = TurnInjectionBridge::new();
    let turn_input_injection_bridge = TurnInputInjectionBridge::new();

    let plugin_factories = plugin_factories_for_surface(
        autonomous,
        execution_mode,
        tavily_key,
        Arc::clone(&instruction_source),
        session_policy.clone(),
        &lash_config,
        host_docs.dir().to_path_buf(),
    );
    let plugin_host = PluginHost::new(plugin_factories).with_dynamic_tools();
    let root_plugins = plugin_host.build_session(
        "root",
        execution_mode,
        session_policy.context_approach.clone(),
        None,
    )?;
    let dynamic_tools = root_plugins
        .dynamic_tools()
        .ok_or_else(|| anyhow::anyhow!("root dynamic tool provider was not initialized"))?;
    attach_mcp_servers(&dynamic_tools, lash_config.mcp_servers())
        .await
        .map_err(|err| anyhow::anyhow!("failed to attach MCP servers: {err}"))?;
    if autonomous {
        apply_autonomous_tool_policy(&dynamic_tools)
            .map_err(|err| anyhow::anyhow!("failed to apply autonomous tool policy: {err}"))?;
    }
    let dynamic_tools_provider: Arc<dyn ToolProvider> = dynamic_tools.clone();
    let toolset_hash = hash12(
        &serde_json::to_vec(&dynamic_tools_provider.definitions())
            .unwrap_or_else(|_| b"[]".to_vec()),
    );
    if args.info {
        let cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| ".".to_string());
        let session_db_path = session_bootstrap.db_path().to_string_lossy().to_string();
        println!(
            "{}",
            info_text(
                &active_provider,
                &model,
                session_policy.model_variant.as_deref(),
                execution_mode,
                &session_policy.context_approach,
                Some(resolved_model_spec.context_window()),
                dynamic_tools_provider.definitions().len(),
                &toolset_hash,
                &cwd,
                None,
                run_session_id.as_deref(),
                Some(&session_db_path),
            )
        );
        return Ok(());
    }
    let initial_model_variant = session_policy.model_variant.clone();
    let session_name = session_bootstrap.session_name();
    let mut logger = session_bootstrap.logger(&model, run_session_id)?;
    let initial_graph = session_bootstrap.initial_graph();
    let services = lash::PersistentRuntimeServices::new_with_bridges(
        root_plugins,
        turn_injection_bridge.clone(),
        turn_input_injection_bridge.clone(),
        store.clone() as Arc<dyn RuntimeStore>,
    );
    let state = PersistedSessionState {
        session_id: "root".to_string(),
        policy: session_policy.clone(),
        session_graph: initial_graph,
        ..PersistedSessionState::default()
    };
    let embedded_host = EmbeddedRuntimeHost::new(host_core).with_session_store_factory(Arc::new(
        DbSessionStoreFactory::new(session_bootstrap.sessions_dir().to_path_buf()),
    ));
    let mut runtime = LashRuntime::from_persistent_background_state(
        session_policy.clone(),
        BackgroundRuntimeHost::new(embedded_host, Arc::new(TokioSessionTaskExecutor::default())),
        services,
        state,
    )
    .await?;
    if let Some(patch) = rlm_globals_patch {
        runtime
            .append_session_nodes(lash::AppendSessionNodesRequest {
                nodes: vec![lash::SessionAppendNode::plugin(
                    lash::INTERNAL_RLM_GLOBALS_PATCH_PLUGIN_TYPE,
                    serde_json::to_value(patch).unwrap_or(serde_json::Value::Null),
                )],
                requires_ancestor_node_id: None,
            })
            .await
            .map_err(|err| anyhow::anyhow!("failed to apply RLM globals patch: {err}"))?;
    }

    // ── Autonomous preset: skip TUI, run session, print response to stdout ──
    if let Some(prompt) = args.print_prompt {
        return run_autonomous(
            runtime,
            prompt,
            SkillCatalog::from_dirs(&crate::paths::default_skill_dirs()),
            AutonomousPersistenceContext {
                await_background_work: args.await_background_work,
                turn_usage_json: args.turn_usage_json.clone(),
            },
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
        runtime,
        plugin_host,
        Arc::clone(&dynamic_tools),
        turn_injection_bridge,
        turn_input_injection_bridge,
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
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyToolProvider;

    fn dummy_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: format!("{name} description"),
            params: Vec::new(),
            returns: "null".to_string(),
            examples: Vec::new(),
            availability: lash::ToolAvailabilityConfig::callable(),
            activation: lash::ToolActivation::Always,
            availability_override: None,
            input_schema_override: None,
            output_schema_override: None,
            execution_mode: ToolExecutionMode::Parallel,
        }
    }

    #[async_trait::async_trait]
    impl ToolProvider for DummyToolProvider {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![
                dummy_tool("read_file"),
                dummy_tool("ask"),
                dummy_tool("plan_exit"),
                dummy_tool("showcase"),
            ]
        }

        async fn execute(&self, name: &str, _args: &serde_json::Value) -> ToolResult {
            ToolResult::err_fmt(format_args!("unexpected tool call: {name}"))
        }
    }

    #[test]
    fn autonomous_policy_disables_interactive_tools() {
        let dynamic_tools =
            DynamicToolProvider::from_tool_provider(Arc::new(DummyToolProvider)).unwrap();

        apply_autonomous_tool_policy(&dynamic_tools).unwrap();

        let tools = dynamic_tools.export_state().tools;
        assert!(tools.contains_key("read_file"));
        assert!(!tools.contains_key("ask"));
        assert!(!tools.contains_key("plan_exit"));
        assert!(!tools.contains_key("showcase"));
    }
}
