//! The single CLI startup pipeline.
//!
//! [`run`] executes the stages in order:
//!
//! 1. preflight early-exits (`--reset`, `--check-update`, `--update`)
//! 2. resolve config + provider credentials (interactive onboarding only
//!    when no stored/shortcut credential exists) — [`provider`]
//! 3. resolve the startup model against the models.dev catalog
//! 4. open the session bootstrap (store + sidecars) exactly once
//! 5. resolve execution mode / context approach from flags + persisted host
//!    config
//! 6. assemble the plugin stack and prompt layer — [`plugins`]
//! 7. build the `LashCore` and open/resume the session — [`session`]
//! 8. hand off to the autonomous runner or the interactive TUI

pub(crate) mod onboarding;
mod plugins;
mod preflight;
mod provider;
pub(crate) mod session;

#[cfg(all(test, feature = "test-provider"))]
pub(crate) use plugins::cli_prompt_config;

use std::collections::BTreeMap;
use std::sync::Arc;

use lash::ModelSpec;
use lash::persistence::FileAttachmentStore;
use lash::tracing::TraceLevel;
use lash_core::SessionPolicy;
use lash_core::ToolProvider;
use lash_core::plugin::{PluginSpec, StaticPluginFactory};
use lash_plugin_mcp::{McpDeferredToolProvider, McpPluginFactory, McpToolProvider};
use lash_tui::Terminal;
use serde_json::Value as JsonValue;

use crate::autonomous::{AutonomousMode, AutonomousPersistenceContext, run_autonomous};
use crate::config::LashConfig;
use crate::execution_settings::{
    ExecutionMode, OmOverrides, RlmTerminationMode, default_rlm_termination_for_mode,
    ensure_supported_execution_mode, parse_execution_mode, parse_standard_context_approach,
};
use crate::info::{info_text, info_text_unconfigured};
use crate::instructions::{FsInstructionSource, InstructionLoaderConfig};
use crate::interactive::run_app;
use crate::model_catalog::{CachedModelCatalog, ResolvedModelSpec};
use crate::model_selection::{
    expose_provider_thinking, models_dev_catalog, parse_model_selection, resolve_model_selection,
    resolve_model_variant, validate_model_selection,
};
use crate::prompt_context_plugin::InstructionSource;
use crate::prompt_tool::CliPromptBridge;
use crate::util::hash12;
use crate::{Args, SkillCatalog, cleanup_terminal};

use self::session::{CliSessionOpener, SessionBootstrap, SessionBootstrapSource};

fn log_filter_enables_debug(filter: &str) -> bool {
    let lowered = filter.to_ascii_lowercase();
    lowered.contains("debug") || lowered.contains("trace")
}

pub(crate) fn effective_lash_log_filter(debug_flag: bool) -> String {
    let env_filter = std::env::var("LASH_LOG")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    match (debug_flag, env_filter) {
        (true, Some(filter)) if log_filter_enables_debug(&filter) => filter,
        (true, _) => "debug".to_string(),
        (false, Some(filter)) => filter,
        (false, None) => "warn".to_string(),
    }
}

pub(crate) fn detailed_debug_logging_enabled(debug_flag: bool) -> bool {
    log_filter_enables_debug(&effective_lash_log_filter(debug_flag))
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
) -> Result<Option<lash_protocol_rlm::RlmProjectedBindings>, String> {
    let mut bindings = lash_protocol_rlm::RlmProjectedBindings::new();
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

fn resolve_rlm_termination(
    args: &Args,
    execution_mode: ExecutionMode,
    persisted_host_config: Option<&session::CliSessionHostConfig>,
) -> Result<Option<RlmTerminationMode>, String> {
    if args.rlm_termination.is_some() && !execution_mode.is_rlm() {
        return Err("`--rlm-termination` requires `--execution-mode rlm`.".to_string());
    }
    if !execution_mode.is_rlm() {
        return Ok(None);
    }
    Ok(args
        .rlm_termination
        .or_else(|| persisted_host_config.and_then(|config| config.rlm_termination))
        .or_else(|| default_rlm_termination_for_mode(execution_mode)))
}

fn resolve_agent_model_specs(
    provider: &lash::provider::ProviderHandle,
    model_catalog: &CachedModelCatalog,
    agent_models: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, ModelSpec>, String> {
    agent_models
        .iter()
        .map(|(capability, configured_model)| {
            let selection = parse_model_selection(configured_model)?;
            validate_model_selection(provider, &selection)?;
            let resolved = resolve_model_selection(provider, &selection, model_catalog)?;
            let variant = resolve_model_variant(provider, &selection.model, None)?;
            let model_spec = resolved.into_model_spec(provider.kind(), variant)?;
            Ok((capability.clone(), model_spec))
        })
        .collect()
}

/// The model the session starts with, resolved from `--model`/`--variant`,
/// the per-provider configured default, and the provider's built-in default.
struct StartupModel {
    model: String,
    variant: Option<String>,
    model_spec: ModelSpec,
    resolved_spec: ResolvedModelSpec,
    agent_model_specs: BTreeMap<String, ModelSpec>,
}

fn resolve_startup_model(
    args: &Args,
    lash_config: &LashConfig,
    provider: &lash::provider::ProviderHandle,
    model_catalog: &CachedModelCatalog,
) -> anyhow::Result<StartupModel> {
    let configured_model_default = lash_config
        .model_default(provider.kind())
        .filter(|selection| !selection.model.trim().is_empty());
    let requested_model = match args
        .model
        .clone()
        .or_else(|| configured_model_default.map(|selection| selection.model.clone()))
    {
        Some(model) => model,
        None => crate::provider_metadata::default_model_for_provider(provider.kind())
            .map_err(anyhow::Error::msg)?
            .to_string(),
    };
    let selection = parse_model_selection(&requested_model).map_err(anyhow::Error::msg)?;
    validate_model_selection(provider, &selection).map_err(anyhow::Error::msg)?;
    let resolved_spec =
        resolve_model_selection(provider, &selection, model_catalog).map_err(anyhow::Error::msg)?;
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
    let variant =
        resolve_model_variant(provider, &model, requested_variant).map_err(anyhow::Error::msg)?;
    let model_spec = resolved_spec
        .clone()
        .into_model_spec(provider.kind(), variant.clone())
        .map_err(anyhow::Error::msg)?;
    let agent_model_specs =
        resolve_agent_model_specs(provider, model_catalog, &lash_config.agent_models)
            .map_err(anyhow::Error::msg)?;
    Ok(StartupModel {
        model,
        variant,
        model_spec,
        resolved_spec,
        agent_model_specs,
    })
}

pub(crate) async fn run(args: Args) -> anyhow::Result<()> {
    // ── Stage 0: flags that exit before any provider/session work ──
    if preflight::handle_early_exit_flags(&args).await? {
        return Ok(());
    }

    // ── Stage 1: resolve config + provider credentials ──
    let config_path = crate::paths::config_file();
    let config_load = LashConfig::load_outcome(&config_path);
    let existing_config = config_load.loaded().cloned();
    let shortcut_api_key = provider::shortcut_api_key(&args);
    if args.info && existing_config.is_none() && shortcut_api_key.is_none() {
        let execution_mode =
            ensure_supported_execution_mode(match args.execution_mode.as_deref() {
                Some(raw) => parse_execution_mode(raw).map_err(anyhow::Error::msg)?,
                None => ExecutionMode::Standard,
            })
            .map_err(anyhow::Error::msg)?;
        resolve_rlm_termination(&args, execution_mode, None).map_err(anyhow::Error::msg)?;
        let cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| ".".to_string());
        if let Some(status) = config_load.status_line() {
            println!("{status}");
        }
        println!("{}", info_text_unconfigured(&execution_mode, &cwd));
        return Ok(());
    }
    if args.mode != crate::CliMode::Interactive && args.debug_ui_trace.is_some() {
        return Err(anyhow::anyhow!(
            "`--debug-ui-trace` only applies to the interactive TUI."
        ));
    }
    if args.mode == crate::CliMode::Json && args.print_prompt.is_none() {
        return Err(anyhow::anyhow!(
            "`--mode json` requires `--print <prompt>`."
        ));
    }
    if args.mode == crate::CliMode::Rpc && args.print_prompt.is_some() {
        return Err(anyhow::anyhow!(
            "`--mode rpc` reads prompts from stdin; do not pass `--print`."
        ));
    }
    if args.mode == crate::CliMode::Rpc && args.turn_usage_json.is_some() {
        return Err(anyhow::anyhow!(
            "`--turn-usage-json` is a single-turn --print artifact and is not supported with `--mode rpc`."
        ));
    }
    let interactive_startup =
        !args.info && args.print_prompt.is_none() && args.mode != crate::CliMode::Rpc;
    let startup_system_message: Option<String> = None;
    let (mut lash_config, mut active_provider) = provider::resolve_config_and_provider(
        &args,
        &config_path,
        config_load,
        shortcut_api_key.as_deref(),
        interactive_startup,
    )
    .await?;
    crate::theme::set_active_theme(lash_config.theme);

    // CLI env/flags override stored config
    if let Some(ref key) = args.tavily_api_key {
        lash_config.set_tavily_api_key(Some(key.clone()));
    }
    if args.print_prompt.is_none() && args.mode != crate::CliMode::Rpc {
        expose_provider_thinking(&mut active_provider);
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

    // ── Stage 2: resolve the startup model ──
    let model_catalog = models_dev_catalog().map_err(anyhow::Error::msg)?;
    if let Err(err) = model_catalog
        .refresh_if_stale(crate::model_catalog::DEFAULT_REFRESH_INTERVAL)
        .await
    {
        eprintln!("warning: failed to refresh models.dev catalog: {err}");
    }
    let startup_model = resolve_startup_model(
        &args,
        &lash_config,
        &active_provider,
        model_catalog.as_ref(),
    )?;
    if args.resume.is_none()
        && args.print_prompt.is_none()
        && args.mode != crate::CliMode::Rpc
        && (args.model.is_some() || args.variant.is_some())
    {
        lash_config.set_model_default(
            active_provider.kind(),
            startup_model.model.clone(),
            startup_model.variant.clone(),
        );
        lash_config.save(&crate::paths::config_file())?;
    }
    let trace_level = TraceLevel::from(args.trace_level);
    let trace_path = if detailed_debug_logging_enabled(args.debug) || trace_level.is_extended() {
        let dir = crate::paths::lash_home().join("sessions");
        Some(dir.join(format!(
            "{}.trace.jsonl",
            chrono::Local::now().format("%Y%m%d_%H%M%S")
        )))
    } else {
        None
    };

    let autonomous = args.print_prompt.is_some() || args.mode == crate::CliMode::Rpc;
    if !autonomous && (args.await_background_work || args.turn_usage_json.is_some()) {
        return Err(anyhow::anyhow!(
            "`--await-background-work` and `--turn-usage-json` require `--print`."
        ));
    }

    // ── Stage 3: open the session bootstrap (the only open; `--info` never
    // touches session storage) ──
    let session_bootstrap = if args.info {
        None
    } else {
        Some(
            SessionBootstrap::open(
                SessionBootstrapSource::from_resume_arg(args.resume.clone()).await,
            )
            .await?,
        )
    };
    if let Some(session_bootstrap) = session_bootstrap.as_ref() {
        tracing::debug!(
            session_file = session_bootstrap.filename(),
            resumed = args.resume.is_some(),
            trace_path = ?trace_path,
            debug_logging = detailed_debug_logging_enabled(args.debug),
            "prepared session bootstrap"
        );
    }
    // Autonomous runs still need a stable session id so provider-side prompt caching
    // and benchmark accounting can key repeated requests within the same session.
    let run_session_id = session_bootstrap
        .as_ref()
        .and_then(|bootstrap| bootstrap.run_session_id());
    let persisted_host_config = session_bootstrap
        .as_ref()
        .and_then(|bootstrap| bootstrap.persisted_host_config());

    // ── Stage 4: execution mode + context approach. CLI flag wins, then
    // persisted host config, then Standard. Resolved BEFORE building the
    // plugin host + runtime so the lashlang thread starts correctly and
    // plugins see the right mode on resume. ──
    let execution_mode = match args.execution_mode.as_deref() {
        Some(raw) => {
            ensure_supported_execution_mode(parse_execution_mode(raw).map_err(anyhow::Error::msg)?)
                .map_err(anyhow::Error::msg)?
        }
        None => persisted_host_config
            .as_ref()
            .map(|config| config.execution_mode)
            .and_then(|mode| ensure_supported_execution_mode(mode).ok())
            .unwrap_or(ExecutionMode::Standard),
    };
    let om_overrides = OmOverrides::from_args(&args);
    if !execution_mode.is_standard()
        && (args.standard_context_approach.is_some() || !om_overrides.is_empty())
    {
        return Err(anyhow::anyhow!(
            "`--context-approach` and OM tuning flags only apply to `--execution-mode standard`."
        ));
    }
    let configured_standard_context_approach = if execution_mode.is_standard() {
        let approach = match args.standard_context_approach.as_deref() {
            Some(raw) => parse_standard_context_approach(raw).map_err(anyhow::Error::msg)?,
            None => persisted_host_config
                .as_ref()
                .and_then(|config| config.standard_context_approach.clone())
                .unwrap_or_default(),
        };
        Some(om_overrides.apply(approach).map_err(anyhow::Error::msg)?)
    } else {
        None
    };
    let rlm_projected_bindings =
        resolve_rlm_projected_bindings(&args).map_err(anyhow::Error::msg)?;
    if rlm_projected_bindings.is_some() && !execution_mode.is_rlm() {
        return Err(anyhow::anyhow!(
            "`--rlm-var` and `--rlm-vars-file` require `--execution-mode rlm`."
        ));
    }
    let rlm_termination =
        resolve_rlm_termination(&args, execution_mode, persisted_host_config.as_ref())
            .map_err(anyhow::Error::msg)?;
    let resolved_host_config = session::CliSessionHostConfig::new(
        execution_mode,
        configured_standard_context_approach.clone(),
        rlm_termination,
    );

    // ── Stage 5: plugin stack + prompt layer + session policy ──
    let instruction_source: Arc<dyn InstructionSource> =
        Arc::new(FsInstructionSource::with_config(InstructionLoaderConfig {
            global_root: Some(crate::paths::lash_home()),
            ..Default::default()
        }));
    let prompt_layer = plugins::cli_prompt_config(autonomous, &execution_mode);
    let session_policy = SessionPolicy {
        model: startup_model.model_spec.clone(),
        provider_id: active_provider.kind().to_string(),
        session_id: run_session_id.clone(),
        autonomous,
        prompt: prompt_layer.clone(),
        ..Default::default()
    };
    let tavily_key = lash_config.tavily_api_key().unwrap_or_default().to_string();
    let prompt_bridge = CliPromptBridge::default();

    let mut plugin_stack =
        plugins::plugin_factories_for_surface(plugins::PluginFactorySurfaceInput {
            autonomous,
            execution_mode,
            standard_context_approach: configured_standard_context_approach.clone(),
            tavily_key,
            instruction_source: Arc::clone(&instruction_source),
            agent_model_specs: &startup_model.agent_model_specs,
            host_docs_dir: host_docs.as_ref().map(|docs| docs.dir().to_path_buf()),
            prompt_bridge: prompt_bridge.clone(),
        });
    // Only configuration errors fail here; an unreachable server keeps
    // reconnecting in the background and its tools appear once it comes up.
    let mcp_factory = Arc::new(
        McpPluginFactory::new(lash_config.mcp_servers().clone())
            .await
            .map_err(|err| anyhow::anyhow!("invalid MCP server configuration: {err}"))?,
    );
    // Reference MCP deferred-resolution wiring (RLM only): resolve a Lashlang
    // call-path absent from the resident catalog into an MCP Tool Grant on
    // demand. Built from the MCP pool's currently enumerated tools.
    let mcp_cataloged_tools = {
        let provider = McpToolProvider::new(Arc::clone(mcp_factory.pool()));
        crate::examples::mcp_discovery::mcp_cataloged_tools("mcp", &provider)
    };
    let has_deferred_mcp_catalog = execution_mode.is_rlm() && !mcp_cataloged_tools.is_empty();
    let mcp_catalog_records = if has_deferred_mcp_catalog {
        crate::examples::mcp_discovery::mcp_catalog_records(&mcp_cataloged_tools)
    } else {
        Default::default()
    };
    let mcp_deferred_resolver: Option<lash::tools::SharedDeferredToolResolver> =
        if has_deferred_mcp_catalog {
            Some(Arc::new(
                crate::examples::mcp_discovery::McpDeferredToolResolver::new(mcp_cataloged_tools),
            ))
        } else {
            None
        };
    // Reference MCP tool-discovery example: install the host-owned
    // `search_tools` overlay only when there is a deferred MCP tail to search.
    // Resident catalog members already render through the protocol-native path.
    if execution_mode.is_rlm() {
        if has_deferred_mcp_catalog {
            plugin_stack.push(Arc::new(
                crate::examples::mcp_discovery::ToolDiscoveryPluginFactory::with_catalog(
                    mcp_catalog_records,
                ),
            ));
            plugin_stack.push(Arc::new(StaticPluginFactory::new(
                "mcp",
                PluginSpec::new().with_tool_provider(Arc::new(McpDeferredToolProvider::new(
                    Arc::clone(mcp_factory.pool()),
                )) as Arc<dyn ToolProvider>),
            )));
        }
    } else {
        plugin_stack.push(mcp_factory);
    }
    if args.info {
        let cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| ".".to_string());
        println!(
            "{}",
            info_text(
                &active_provider,
                &startup_model.model,
                session_policy.model.variant.effort(),
                &execution_mode,
                configured_standard_context_approach.as_ref(),
                Some(startup_model.resolved_spec.context_window()),
                None,
                &cwd,
                None,
                None,
                None,
            )
        );
        return Ok(());
    }

    // ── Stage 6: build LashCore + open/resume the session ──
    std::fs::create_dir_all(crate::paths::lash_home())?;
    let runtime_factory = CliSessionOpener::new(
        plugin_stack.clone(),
        prompt_layer,
        Arc::new(FileAttachmentStore::new(crate::paths::attachments_dir())),
        active_provider.clone(),
        mcp_deferred_resolver,
        trace_path,
        trace_level,
    );
    let session_bootstrap =
        session_bootstrap.expect("session bootstrap is opened for every non --info run");
    let opened_session = runtime_factory
        .open_prepared(
            session_bootstrap,
            session_policy.clone(),
            resolved_host_config,
        )
        .await?;
    let session = opened_session.session;

    // Best-effort attachment GC: reclaim content-addressed blobs that no live
    // session references. This CLI install owns its blob directory exclusively,
    // so a blob with no manifest ref across any session database is garbage. A
    // generous grace period protects in-flight writes. The sweep runs in the
    // background so it never delays an interactive session.
    {
        let attachments_dir = crate::paths::attachments_dir();
        let sessions_dir = crate::session_log::sessions_dir();
        tokio::spawn(async move {
            // One hour: far larger than any live turn, so an in-flight put whose
            // intent has not yet been observed is never swept.
            const ATTACHMENT_GC_GRACE_MS: u64 = 60 * 60 * 1000;
            let backend = FileAttachmentStore::new(attachments_dir);
            let root_set = lash_sqlite_store::SqliteSessionStoreFactory::new(sessions_dir);
            match lash::persistence::reclaim_unreferenced_attachments(
                &root_set,
                &backend,
                ATTACHMENT_GC_GRACE_MS,
            )
            .await
            {
                Ok(report) => {
                    if report.reclaimed_count > 0 || !report.failed_ids.is_empty() {
                        tracing::info!(
                            scanned = report.scanned_blob_count,
                            reclaimed = report.reclaimed_count,
                            failed = report.failed_ids.len(),
                            "attachment gc sweep reclaimed unreferenced blobs"
                        );
                    }
                }
                Err(err) => {
                    tracing::warn!(error = %err, "attachment gc sweep failed");
                }
            }
        });
    }

    if autonomous {
        plugins::apply_autonomous_tool_policy(&session)
            .await
            .map_err(|err| anyhow::anyhow!("failed to apply autonomous tool policy: {err}"))?;
    }
    let active_tool_definitions = session.admin().tools().active_manifests().await?;
    let toolset_hash =
        hash12(&serde_json::to_vec(&active_tool_definitions).unwrap_or_else(|_| b"[]".to_vec()));
    let initial_policy = session.policy_snapshot();
    let initial_model_variant = initial_policy.model.variant.clone();
    let store = opened_session.bootstrap.store();
    let session_name = opened_session.bootstrap.session_name();
    let mut logger = opened_session.logger;
    session
        .admin()
        .commands()
        .refresh_tool_catalog("bootstrap", "bootstrap-refresh-tool-catalog")
        .await?;
    if rlm_projected_bindings.is_some() && args.print_prompt.is_none() {
        return Err(anyhow::anyhow!(
            "`--rlm-var` and `--rlm-vars-file` are currently supported for autonomous `--print-prompt` turns; interactive hosts should use the RLM projection API."
        ));
    }

    // ── Stage 7a: autonomous preset — skip TUI, run session, print response ──
    if let Some(prompt) = args.print_prompt.clone() {
        let mode = match args.mode {
            crate::CliMode::Interactive => AutonomousMode::Print,
            crate::CliMode::Json => AutonomousMode::Json,
            crate::CliMode::Rpc => AutonomousMode::Rpc,
        };
        return run_autonomous(
            session,
            prompt,
            SkillCatalog::from_dirs(&crate::paths::default_skill_dirs()),
            AutonomousPersistenceContext {
                await_background_work: args.await_background_work,
                turn_usage_json: args.turn_usage_json.clone(),
            },
            rlm_projected_bindings,
            mode,
        )
        .await;
    }
    if args.mode == crate::CliMode::Rpc {
        return run_autonomous(
            session,
            String::new(),
            SkillCatalog::from_dirs(&crate::paths::default_skill_dirs()),
            AutonomousPersistenceContext {
                await_background_work: args.await_background_work,
                turn_usage_json: args.turn_usage_json.clone(),
            },
            None,
            AutonomousMode::Rpc,
        )
        .await;
    }

    // ── Stage 7b: interactive TUI ──
    // Install panic hook that restores the terminal. Tokio worker-task panics
    // also run this hook; those must not tear down the active alternate screen.
    let terminal_owner_thread = std::thread::current().id();
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if should_restore_terminal_for_panic(terminal_owner_thread) {
            cleanup_terminal();
            default_hook(info);
        } else {
            tracing::error!(
                thread = ?std::thread::current().name(),
                panic = %info,
                "background task panicked"
            );
        }
    }));

    // Initialize terminal
    crate::util::require_interactive_terminal("the lash TUI")?;
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
        startup_model.model,
        startup_model.resolved_spec.context_window(),
        session_name,
        model_catalog,
        Arc::clone(&store),
        toolset_hash,
        crate::model_selection::variant_from_reasoning_selection(initial_model_variant),
        execution_mode,
        startup_system_message,
    )
    .await
}

fn should_restore_terminal_for_panic(terminal_owner_thread: std::thread::ThreadId) -> bool {
    std::thread::current().id() == terminal_owner_thread
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::env_lock;
    use clap::Parser as _;

    #[test]
    fn terminal_panic_cleanup_is_limited_to_owner_thread() {
        let owner = std::thread::current().id();
        assert!(should_restore_terminal_for_panic(owner));

        std::thread::spawn(move || {
            assert!(!should_restore_terminal_for_panic(owner));
        })
        .join()
        .expect("thread should not panic");
    }

    #[test]
    fn effective_lash_log_filter_defaults_and_promotes_debug_flag() {
        let _env_guard = env_lock().blocking_lock();
        unsafe { std::env::remove_var("LASH_LOG") };
        assert_eq!(effective_lash_log_filter(false), "warn");
        assert_eq!(effective_lash_log_filter(true), "debug");

        unsafe { std::env::set_var("LASH_LOG", "trace") };
        assert_eq!(effective_lash_log_filter(false), "trace");
        assert_eq!(effective_lash_log_filter(true), "trace");

        unsafe { std::env::set_var("LASH_LOG", "warn") };
        assert_eq!(effective_lash_log_filter(true), "debug");
    }

    #[test]
    fn rlm_termination_defaults_to_natural_in_rlm_mode() {
        let args =
            crate::Args::try_parse_from(["lash", "--execution-mode", "rlm"]).expect("parse args");

        assert_eq!(
            resolve_rlm_termination(&args, ExecutionMode::Rlm, None).expect("termination"),
            Some(RlmTerminationMode::Natural)
        );
    }

    #[test]
    fn rlm_termination_honors_explicit_finish_required() {
        let args = crate::Args::try_parse_from([
            "lash",
            "--execution-mode",
            "rlm",
            "--rlm-termination",
            "finish-required",
        ])
        .expect("parse args");

        assert_eq!(
            resolve_rlm_termination(&args, ExecutionMode::Rlm, None).expect("termination"),
            Some(RlmTerminationMode::FinishRequired)
        );
    }

    #[test]
    fn rlm_termination_uses_persisted_value_when_flag_omitted() {
        let args =
            crate::Args::try_parse_from(["lash", "--execution-mode", "rlm"]).expect("parse args");
        let persisted = session::CliSessionHostConfig::new(
            ExecutionMode::Rlm,
            None,
            Some(RlmTerminationMode::FinishRequired),
        );

        assert_eq!(
            resolve_rlm_termination(&args, ExecutionMode::Rlm, Some(&persisted))
                .expect("termination"),
            Some(RlmTerminationMode::FinishRequired)
        );
    }

    #[test]
    fn rlm_termination_flag_requires_rlm_mode() {
        let args = crate::Args::try_parse_from(["lash", "--rlm-termination", "natural"])
            .expect("parse args");

        let err = resolve_rlm_termination(&args, ExecutionMode::Standard, None)
            .expect_err("standard mode should reject rlm termination flag");

        assert!(err.contains("--execution-mode rlm"));
    }
}
