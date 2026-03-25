use std::sync::Arc;

use lash::provider::{LashConfig, Provider};
use lash::tools::{AgentCallConfig, DefaultToolPluginDeps};
use lash::*;

use crate::autonomous::{AutonomousPersistenceContext, run_autonomous};
use crate::interactive::{generate_session_name, run_app};
use crate::session_log::{self, SessionLogger};
use crate::{Args, fork, setup};
use crate::{
    autonomous_prompt_overrides, cleanup_terminal, configure_terminal_ui,
    ensure_supported_execution_mode, hash12, info_text, info_text_unconfigured, models_dev_catalog,
    parse_context_strategy, parse_execution_mode, parse_model_selection, resolve_context_strategy,
    resolve_model_selection, resolve_model_variant, validate_model_selection,
};

pub(crate) async fn run(
    args: Args,
    mut prompt_overrides: Vec<PromptSectionOverride>,
) -> anyhow::Result<()> {
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

        let lash_dir = lash::lash_home();
        let cache_dir = lash::lash_cache_dir();

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

    // Resolve config before TUI init (may need interactive terminal)
    let existing_config = LashConfig::load();
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
    let mut lash_config = if args.provider || existing_config.is_none() {
        if let Some(ref key) = args.api_key {
            // Shortcut: env var or --api-key activates OpenAI-generic directly.
            let provider = Provider::OpenAiGeneric {
                api_key: key.clone(),
                base_url: args.base_url.clone(),
                options: lash::provider::ProviderOptions::default(),
            };
            let mut cfg = existing_config
                .clone()
                .unwrap_or_else(|| LashConfig::new(provider.clone()));
            cfg.upsert_provider(provider.clone());
            let _ = cfg.set_active_provider_kind(provider.kind());
            cfg.set_tavily_api_key(args.tavily_api_key.clone());
            cfg
        } else {
            setup::run_setup_with_existing(existing_config.as_ref()).await?
        }
    } else {
        // SAFETY: else branch means existing_config.is_some() (checked above)
        #[allow(clippy::unnecessary_unwrap)]
        let mut c = existing_config.unwrap();
        match c.active_provider_mut().ensure_fresh().await {
            Ok(true) => c.save()?, // persist refreshed tokens
            Ok(false) => {}
            Err(err) => {
                if interactive_startup {
                    let provider_label = c.active_provider().label();
                    startup_system_message = Some(format!(
                        "{} refresh failed: {}. Lash opened in recovery mode. Use /provider or relaunch with --provider to reauthenticate or switch providers.",
                        provider_label, err
                    ));
                } else {
                    return Err(err.into());
                }
            }
        }
        c
    };

    // CLI env/flags override stored config
    if let Some(ref key) = args.tavily_api_key {
        lash_config.set_tavily_api_key(Some(key.clone()));
    }
    if let Some(models) = lash_config.agent_models.as_mut()
        && models.recall_agent.is_none()
    {
        models.recall_agent = models.low.clone();
    }
    let requested_context_strategy = match args.context_strategy.as_deref() {
        Some(raw) => Some(parse_context_strategy(raw).map_err(anyhow::Error::msg)?),
        None => None,
    };
    let context_strategy =
        resolve_context_strategy(lash_config.context_strategy(), requested_context_strategy)
            .map_err(anyhow::Error::msg)?;
    lash_config.set_context_strategy(context_strategy);
    if args.print_prompt.is_none() {
        lash_config.save()?;
    }
    let model_catalog = models_dev_catalog().map_err(anyhow::Error::msg)?;
    if let Err(err) = model_catalog
        .refresh_if_stale(lash::model_info::DEFAULT_REFRESH_INTERVAL)
        .await
    {
        eprintln!("warning: failed to refresh models.dev catalog: {err}");
    }

    let requested_model = args
        .model
        .clone()
        .unwrap_or_else(|| lash_config.active_provider().default_model().to_string());
    let selection = parse_model_selection(&requested_model).map_err(anyhow::Error::msg)?;
    validate_model_selection(lash_config.active_provider(), &selection)
        .map_err(anyhow::Error::msg)?;
    let resolved_model_spec = resolve_model_selection(
        lash_config.active_provider(),
        &selection,
        model_catalog.as_ref(),
    )
    .map_err(anyhow::Error::msg)?;
    let model = selection.model.clone();
    let model_variant = resolve_model_variant(
        lash_config.active_provider(),
        &model,
        args.variant.as_deref(),
    )
    .map_err(anyhow::Error::msg)?;
    let execution_mode = ensure_supported_execution_mode(match args.execution_mode.as_deref() {
        Some(raw) => parse_execution_mode(raw).map_err(anyhow::Error::msg)?,
        None => ExecutionMode::Standard,
    })
    .map_err(anyhow::Error::msg)?;

    let llm_log_path = std::env::var("LASH_LOG").ok().and_then(|level| {
        let l = level.to_lowercase();
        if l == "debug" || l == "trace" || l.contains("debug") || l.contains("trace") {
            let dir = lash::lash_home().join("sessions");
            Some(dir.join(format!(
                "{}.llm.jsonl",
                chrono::Local::now().format("%Y%m%d_%H%M%S")
            )))
        } else {
            None
        }
    });

    let sessions_dir = lash::lash_home().join("sessions");
    std::fs::create_dir_all(&sessions_dir)?;
    let session_filename = args
        .resume
        .clone()
        .unwrap_or_else(session_log::new_session_filename);
    let resume_start = if args.resume.is_some() {
        Some(session_log::load_session_start(&session_filename)?)
    } else {
        None
    };

    let autonomous = args.print_prompt.is_some();
    if autonomous {
        prompt_overrides.extend(autonomous_prompt_overrides());
    }
    // Autonomous runs still need a stable session id so provider-side prompt caching
    // and benchmark accounting can key repeated requests within the same session.
    let run_session_id = resume_start
        .as_ref()
        .map(|start| start.session_id.clone())
        .or_else(|| Some(uuid::Uuid::new_v4().to_string()));
    let instruction_source: Arc<dyn InstructionSource> = Arc::new(FsInstructionSource::new());
    let session_policy = SessionPolicy {
        model: model.clone(),
        provider: lash_config.active_provider().clone(),
        model_variant,
        recall_agent_model: lash_config
            .agent_models
            .as_ref()
            .and_then(|models| models.recall_agent.clone()),
        max_context_tokens: Some(resolved_model_spec.context_window() as usize),
        session_id: run_session_id.clone(),
        execution_mode,
        context_strategy,
        ..Default::default()
    };
    let host_config = RuntimeHostConfig {
        prompt_renderer: default_prompt_renderer(),
        prompt_overrides,
        llm_log_path,
        ..RuntimeHostConfig::default()
    };

    // Build store (SQLite-backed archive)
    let db_path = sessions_dir.join(&session_filename);
    let store = Arc::new(Store::open(&db_path)?);

    let tavily_key = lash_config.tavily_api_key().unwrap_or_default().to_string();
    let prompt_bridge = PromptBridge::new();
    let turn_injection_bridge = TurnInjectionBridge::new();

    let agent_call_config = AgentCallConfig {
        low_tier_execution_mode: lash_config
            .runtime
            .low_tier_subagent_execution_mode
            .unwrap_or(ExecutionMode::Standard),
    };

    let mut plugin_factories = default_tool_plugin_factories(
        execution_mode,
        DefaultToolPluginDeps {
            store: Some(Arc::clone(&store)),
            tavily_api_key: if tavily_key.is_empty() {
                None
            } else {
                Some(tavily_key)
            },
            prompt_bridge: (!autonomous).then_some(prompt_bridge.clone()),
            instruction_source: Some(Arc::clone(&instruction_source)),
        },
    );
    plugin_factories.extend(vec![
        Arc::new(BuiltinPromptContextPluginFactory::new(
            Arc::clone(&instruction_source),
            PromptContextPluginConfig::default(),
        )) as Arc<dyn PluginFactory>,
        Arc::new(BuiltinPlanTrackerPluginFactory),
        Arc::new(BuiltinPlanModePluginFactory::default()),
        Arc::new(fork::ForkPluginFactory),
        Arc::new(AgentCallPluginFactory::new(
            session_policy.clone(),
            agent_call_config,
            lash_config.agent_models.clone(),
        )),
    ]);
    let plugin_host = PluginHost::new(plugin_factories).with_dynamic_tools();
    let root_plugins = plugin_host.build_session("root", execution_mode, None)?;
    let dynamic_tools = root_plugins
        .dynamic_tools()
        .ok_or_else(|| anyhow::anyhow!("root dynamic tool provider was not initialized"))?;
    attach_mcp_servers(&dynamic_tools, lash_config.mcp_servers())
        .await
        .map_err(|err| anyhow::anyhow!("failed to attach MCP servers: {err}"))?;
    let dynamic_tools_provider: Arc<dyn ToolProvider> = dynamic_tools.clone();
    let toolset_hash = hash12(
        &serde_json::to_vec(&dynamic_tools_provider.definitions())
            .unwrap_or_else(|_| b"[]".to_vec()),
    );
    if args.info {
        let cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| ".".to_string());
        println!(
            "{}",
            info_text(
                lash_config.active_provider(),
                &model,
                session_policy.model_variant.as_deref(),
                execution_mode,
                Some(resolved_model_spec.context_window()),
                dynamic_tools_provider.definitions().len(),
                &toolset_hash,
                context_strategy,
                &cwd,
                None,
            )
        );
        return Ok(());
    }
    let initial_model_variant = session_policy.model_variant.clone();
    let runtime = LashRuntime::from_state(
        session_policy.clone(),
        host_config,
        RuntimeServices::new_with_bridges(
            root_plugins,
            prompt_bridge,
            turn_injection_bridge.clone(),
        )
        .with_store(Arc::clone(&store)),
        AgentStateEnvelope {
            agent_id: "root".to_string(),
            policy: session_policy.clone(),
            ..AgentStateEnvelope::default()
        },
    )
    .await?;

    // ── Autonomous preset: skip TUI, run agent, print response to stdout ──
    if let Some(prompt) = args.print_prompt {
        return run_autonomous(
            runtime,
            prompt,
            SkillCatalog::load(),
            AutonomousPersistenceContext {
                store: Arc::clone(&store),
                dynamic_state: dynamic_tools.export_state(),
                provider: session_policy.provider.clone(),
                configured_model: model.clone(),
                context_window: resolved_model_spec.context_window(),
                model_variant: initial_model_variant.clone(),
                toolset_hash: toolset_hash.clone(),
            },
        )
        .await;
    }

    let session_name = resume_start
        .as_ref()
        .map(|start| start.session_name.clone())
        .unwrap_or_else(|| generate_session_name(&sessions_dir));
    let mut logger = if args.resume.is_some() {
        SessionLogger::resume(Arc::clone(&store), &session_filename)?
    } else {
        SessionLogger::new(
            Arc::clone(&store),
            session_filename.clone(),
            &model,
            run_session_id,
            session_name.clone(),
        )?
    };

    // Install panic hook that restores the terminal
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        cleanup_terminal();
        default_hook(info);
    }));

    // Initialize terminal
    let terminal = ratatui::init();

    configure_terminal_ui(args.no_mouse)?;

    let result = run_app(
        terminal,
        runtime,
        plugin_host,
        Arc::clone(&dynamic_tools),
        turn_injection_bridge,
        &mut logger,
        &args,
        lash_config.active_provider().clone(),
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
    .await;

    // Disable focus change tracking
    let _ = crossterm::execute!(std::io::stdout(), crossterm::event::DisableFocusChange);

    // Disable mouse capture
    if !args.no_mouse {
        let _ = crossterm::execute!(std::io::stdout(), crossterm::event::DisableMouseCapture);
    }

    // Pop kitty keyboard protocol
    let _ = crossterm::execute!(
        std::io::stdout(),
        crossterm::event::PopKeyboardEnhancementFlags
    );

    cleanup_terminal();

    result
}
