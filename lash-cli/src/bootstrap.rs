use std::sync::Arc;

use lash::provider::{LashConfig, Provider};
use lash::*;
use lash_tui::Terminal;
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::autonomous::{AutonomousPersistenceContext, run_autonomous};
use crate::delegate_tools::{DelegateToolConfig, DelegateToolsPluginFactory};
use crate::interactive::{generate_session_name, run_app};
use crate::session_log::{self, DbSessionStoreFactory, SessionLogger};
use crate::{Args, setup};
use crate::{
    apply_context_approach_overrides, autonomous_prompt_overrides, cleanup_terminal,
    ensure_supported_execution_mode, hash12, info_text, info_text_unconfigured, models_dev_catalog,
    parse_context_approach, parse_execution_mode, parse_model_selection, resolve_model_selection,
    resolve_model_variant, validate_model_selection,
};

fn plugin_factories_for_surface(
    autonomous: bool,
    execution_mode: ExecutionMode,
    store: Arc<Store>,
    tavily_key: String,
    instruction_source: Arc<dyn InstructionSource>,
    session_policy: SessionPolicy,
    lash_config: &LashConfig,
) -> Vec<Arc<dyn PluginFactory>> {
    let delegate_tool_config = DelegateToolConfig {
        low_tier_execution_mode: lash_config
            .runtime
            .low_tier_delegate_execution_mode
            .unwrap_or(ExecutionMode::Standard),
    };

    let mut plugin_factories = default_tool_plugin_factories(
        execution_mode,
        DefaultToolPluginDeps {
            store: Some(store as Arc<dyn RuntimeStore>),
            tavily_api_key: if tavily_key.is_empty() {
                None
            } else {
                Some(tavily_key)
            },
            enable_user_prompts: !autonomous,
            instruction_source: Some(Arc::clone(&instruction_source)),
        },
    );
    plugin_factories.push(Arc::new(BuiltinPromptContextPluginFactory::new(
        Arc::clone(&instruction_source),
        PromptContextPluginConfig::default(),
    )) as Arc<dyn PluginFactory>);
    if !autonomous {
        plugin_factories.push(Arc::new(BuiltinPlanTrackerPluginFactory));
        plugin_factories.push(Arc::new(BuiltinPlanModePluginFactory::new(
            Default::default(),
        )));
        plugin_factories.push(Arc::new(lash::BuiltinUiActivityPluginFactory));
    }
    plugin_factories.push(Arc::new(crate::rlm_stream_mask::RlmStreamMaskPluginFactory));
    plugin_factories.push(Arc::new(DelegateToolsPluginFactory::new(
        session_policy,
        delegate_tool_config,
        lash_config.agent_models.clone(),
    )));
    plugin_factories
}

fn autonomous_tool_allowed(name: &str) -> bool {
    !matches!(
        name,
        "ask" | "wait" | "show_snippet_to_user" | "update_plan" | "showcase"
    ) && !name.starts_with("plan_")
        && name != "request_user_input"
}

fn apply_autonomous_tool_policy(
    dynamic_tools: &DynamicToolProvider,
) -> Result<(), ReconfigureError> {
    let mut snapshot = dynamic_tools.export_state();
    snapshot
        .enabled_tools
        .retain(|name| autonomous_tool_allowed(name));
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

    if args.check_update {
        println!("{}", crate::update::check_update_text().await?);
        return Ok(());
    }

    if args.update {
        crate::update::install_latest_release().await?;
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
            // Shortcut: env var or --api-key activates OpenAI-compatible directly.
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
    let llm_log_path = if crate::detailed_debug_logging_enabled(args.debug) {
        let dir = lash::lash_home().join("sessions");
        Some(dir.join(format!(
            "{}.llm.jsonl",
            chrono::Local::now().format("%Y%m%d_%H%M%S")
        )))
    } else {
        None
    };

    let sessions_dir = lash::lash_home().join("sessions");
    std::fs::create_dir_all(&sessions_dir)?;
    let session_filename = args
        .resume
        .clone()
        .unwrap_or_else(session_log::new_session_filename);
    tracing::debug!(
        session_file = session_filename,
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
    if autonomous {
        prompt_overrides.extend(autonomous_prompt_overrides());
    }
    // Build store (SQLite-backed archive)
    let db_path = sessions_dir.join(&session_filename);
    let store = Arc::new(Store::open(&db_path)?);
    let resume_start = if args.resume.is_some() {
        store
            .load_session_meta()
            .map(|meta| session_log::SessionStart {
                session_id: meta.session_id,
                session_name: meta.session_name,
            })
    } else {
        None
    };
    // Autonomous runs still need a stable session id so provider-side prompt caching
    // and benchmark accounting can key repeated requests within the same session.
    let run_session_id = resume_start
        .as_ref()
        .map(|start| start.session_id.clone())
        .or_else(|| Some(uuid::Uuid::new_v4().to_string()));

    // Peek the persisted session config so we can use the correct
    // execution mode and context approach BEFORE building the plugin
    // host and runtime. Without this, resuming a Rlm session from a
    // Standard-mode bootstrap would fail to start the lashlang thread
    // and would build plugins with the wrong mode.
    let persisted_session_config = if args.resume.is_some() {
        store
            .load_session_graph()
            .and_then(|graph| graph.latest_session_config())
    } else {
        None
    };

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

    let instruction_source: Arc<dyn InstructionSource> = Arc::new(FsInstructionSource::new());
    let session_policy = SessionPolicy {
        model: model.clone(),
        provider: lash_config.active_provider().clone(),
        model_variant,
        max_context_tokens: Some(resolved_model_spec.context_window() as usize),
        session_id: run_session_id.clone(),
        execution_mode,
        context_approach: configured_context_approach.clone(),
        ..Default::default()
    };
    let host_config = RuntimeHostConfig {
        user_prompts_enabled: !autonomous,
        session_store_factory: Some(Arc::new(DbSessionStoreFactory::new(sessions_dir.clone()))),
        prompt_renderer: default_prompt_renderer(),
        prompt_overrides,
        llm_log_path,
        ..RuntimeHostConfig::default()
    };

    let tavily_key = lash_config.tavily_api_key().unwrap_or_default().to_string();
    let turn_injection_bridge = TurnInjectionBridge::new();

    let plugin_factories = plugin_factories_for_surface(
        autonomous,
        execution_mode,
        Arc::clone(&store),
        tavily_key,
        Arc::clone(&instruction_source),
        session_policy.clone(),
        &lash_config,
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
        println!(
            "{}",
            info_text(
                lash_config.active_provider(),
                &model,
                session_policy.model_variant.as_deref(),
                execution_mode,
                &session_policy.context_approach,
                Some(resolved_model_spec.context_window()),
                dynamic_tools_provider.definitions().len(),
                &toolset_hash,
                &cwd,
                None,
            )
        );
        return Ok(());
    }
    let initial_model_variant = session_policy.model_variant.clone();
    let session_name = resume_start
        .as_ref()
        .map(|start| start.session_name.clone())
        .unwrap_or_else(|| generate_session_name(&sessions_dir));
    let mut logger = if resume_start.is_some() {
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
    let initial_graph = if args.resume.is_some() {
        store.load_session_graph().unwrap_or_default()
    } else {
        lash::SessionGraph::default()
    };
    let mut runtime = LashRuntime::from_state(
        session_policy.clone(),
        host_config,
        RuntimeServices::new_with_bridges(root_plugins, turn_injection_bridge.clone())
            .with_store(store.clone() as Arc<dyn RuntimeStore>),
        SessionStateEnvelope {
            session_id: "root".to_string(),
            policy: session_policy.clone(),
            session_graph: initial_graph,
            ..SessionStateEnvelope::default()
        },
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
            SkillCatalog::load(),
            AutonomousPersistenceContext {
                store: Arc::clone(&store),
                dynamic_state: dynamic_tools.export_state(),
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
            enabled: true,
            injected: false,
            input_schema_override: None,
            output_schema_override: None,
        }
    }

    #[async_trait::async_trait]
    impl ToolProvider for DummyToolProvider {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![
                dummy_tool("read_file"),
                dummy_tool("ask"),
                dummy_tool("update_plan"),
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

        let enabled = dynamic_tools.enabled_tools();
        assert!(enabled.contains("read_file"));
        assert!(!enabled.contains("ask"));
        assert!(!enabled.contains("update_plan"));
        assert!(!enabled.contains("plan_exit"));
        assert!(!enabled.contains("showcase"));
    }
}
