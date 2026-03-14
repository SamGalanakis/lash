mod activity;
mod app;
mod command;
mod event;
mod fork;
mod input_items;
mod markdown;
mod plugin_surface;
mod prompt_overrides;
mod replay;
mod resume;
mod session_log;
mod setup;
mod skill;
mod theme;
mod ui;
mod util;

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::Parser;
use crossterm::cursor::SetCursorStyle;
use crossterm::event::{Event as TermEvent, KeyCode, KeyEventKind, KeyModifiers};
use lash_core::agent::{Message, MessageRole};
use lash_core::provider::{LashConfig, OPENAI_GENERIC_DEFAULT_BASE_URL, Provider, ProviderKind};
use lash_core::tools::{
    AgentCall, AgentCallConfig, FilteredTools, SwitchableTools, ToolSet, ToolSetDeps,
};
use lash_core::*;
use ratatui::DefaultTerminal;
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use app::{App, DisplayBlock, QueuedTurn};
use event::AppEvent;
use input_items::{build_items_from_editor_input, insert_inline_marker};
use prompt_overrides::resolve_prompt_overrides;
use session_log::SessionLogger;

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
const ROOT_SESSION_ID: &str = "root";
const LONG_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "\n",
    "lash-core ",
    env!("CARGO_PKG_VERSION")
);

fn autonomous_prompt_overrides() -> Vec<PromptSectionOverride> {
    vec![
        PromptSectionOverride {
            section: PromptSectionName::Identity,
            mode: PromptOverrideMode::Prepend,
            content: "You are an autonomous AI coding agent running without a human in the loop.\nComplete the task end-to-end without asking for user input.".to_string(),
        },
        PromptSectionOverride {
            section: PromptSectionName::ExecutionContract,
            mode: PromptOverrideMode::Append,
            content: "- No user is available during this run. Do not rely on `ask`; make the best reasonable decision from local context and continue.".to_string(),
        },
    ]
}

#[derive(Parser)]
#[command(name = "lash", version = APP_VERSION, long_version = LONG_VERSION)]
struct Args {
    /// OpenAI-generic API key (optional — use --provider to configure interactively)
    #[arg(long, env = "OPENAI_GENERIC_API_KEY")]
    api_key: Option<String>,

    /// Tavily API key for web search
    #[arg(long, env = "TAVILY_API_KEY")]
    tavily_api_key: Option<String>,

    /// Model name (defaults per provider: Claude/Codex/OpenAI-generic/Google OAuth)
    #[arg(long)]
    model: Option<String>,

    /// Provider-native model variant (for example: high, max, xhigh)
    #[arg(long)]
    variant: Option<String>,

    /// Execution backend (`repl` or `standard`, default: `standard`)
    #[arg(long = "execution-mode")]
    execution_mode: Option<String>,

    /// Soft context-fold watermark percentage
    #[arg(long = "context-fold-soft-pct")]
    context_fold_soft_pct: Option<u8>,

    /// Hard context-fold watermark percentage
    #[arg(long = "context-fold-hard-pct")]
    context_fold_hard_pct: Option<u8>,

    /// Base URL for the LLM API
    #[arg(long, default_value = OPENAI_GENERIC_DEFAULT_BASE_URL)]
    base_url: String,

    /// Disable mouse scroll support (re-enables terminal text selection)
    #[arg(long)]
    no_mouse: bool,

    /// Resume an existing session file on startup
    #[arg(long, value_name = "SESSION.jsonl")]
    resume: Option<String>,

    /// Queue and immediately send a prompt after startup resume
    #[arg(long, value_name = "PROMPT")]
    resume_prompt: Option<String>,

    /// Force re-run provider setup
    #[arg(long)]
    provider: bool,

    /// Delete all lash data (~/.lash/ and ~/.cache/lash/) and exit
    #[arg(long)]
    reset: bool,

    /// Print current runtime/config info and exit
    #[arg(long)]
    info: bool,

    /// Run autonomously: execute prompt, print response to stdout, exit
    #[arg(short = 'p', long = "print")]
    print_prompt: Option<String>,

    /// Replace a prompt section: --prompt-replace section=text
    #[arg(long = "prompt-replace", value_name = "SECTION=TEXT")]
    prompt_replace: Vec<String>,

    /// Replace a prompt section from file: --prompt-replace-file section=path
    #[arg(long = "prompt-replace-file", value_name = "SECTION=PATH")]
    prompt_replace_file: Vec<String>,

    /// Prepend text to a prompt section: --prompt-prepend section=text
    #[arg(long = "prompt-prepend", value_name = "SECTION=TEXT")]
    prompt_prepend: Vec<String>,

    /// Prepend text to a prompt section from file: --prompt-prepend-file section=path
    #[arg(long = "prompt-prepend-file", value_name = "SECTION=PATH")]
    prompt_prepend_file: Vec<String>,

    /// Append text to a prompt section: --prompt-append section=text
    #[arg(long = "prompt-append", value_name = "SECTION=TEXT")]
    prompt_append: Vec<String>,

    /// Append text to a prompt section from file: --prompt-append-file section=path
    #[arg(long = "prompt-append-file", value_name = "SECTION=PATH")]
    prompt_append_file: Vec<String>,

    /// Disable a prompt section entirely.
    #[arg(long = "prompt-disable", value_name = "SECTION")]
    prompt_disable: Vec<String>,
}

fn cleanup_terminal() {
    // Pop kitty keyboard protocol, restore background color and cursor style
    let _ = crossterm::execute!(
        std::io::stdout(),
        crossterm::event::PopKeyboardEnhancementFlags
    );
    let _ = crossterm::execute!(
        std::io::stdout(),
        crossterm::style::Print("\x1b]111\x1b\\"),
        SetCursorStyle::DefaultUserShape
    );
    ratatui::restore();
}

fn configure_terminal_ui(no_mouse: bool) -> anyhow::Result<()> {
    crossterm::execute!(
        std::io::stdout(),
        crossterm::style::Print("\x1b]11;rgb:0e/0d/0b\x1b\\"),
        SetCursorStyle::SteadyBar
    )?;
    let _ = crossterm::execute!(
        std::io::stdout(),
        crossterm::event::PushKeyboardEnhancementFlags(
            crossterm::event::KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
        )
    );
    if !no_mouse {
        crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture)?;
    }
    crossterm::execute!(std::io::stdout(), crossterm::event::EnableFocusChange)?;
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Set up file-based structured tracing (JSON logs at $LASH_HOME/lash.log)
    {
        let log_dir = lash_core::lash_home();
        std::fs::create_dir_all(&log_dir).ok();
        let log_file = std::fs::File::create(log_dir.join("lash.log"))?;

        use tracing_subscriber::EnvFilter;
        let filter = EnvFilter::try_from_env("LASH_LOG").unwrap_or_else(|_| EnvFilter::new("warn"));
        tracing_subscriber::fmt()
            .json()
            .flatten_event(true)
            .with_current_span(true)
            .with_span_list(true)
            .with_env_filter(filter)
            .with_writer(log_file)
            .with_ansi(false)
            .init();
    }

    let args = Args::parse();
    let mut prompt_overrides = resolve_prompt_overrides(&args)?;

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

        let lash_dir = lash_core::lash_home();
        let cache_dir = lash_core::lash_cache_dir();

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
    if args.info
        && existing_config.is_none()
        && args.api_key.is_none()
        && std::env::var("ANTHROPIC_API_KEY").is_err()
    {
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
    let mut lash_config = if args.provider || existing_config.is_none() {
        if let Some(ref key) = args.api_key {
            // Shortcut: env var or --api-key activates OpenAI-generic directly.
            let provider = Provider::OpenAiGeneric {
                api_key: key.clone(),
                base_url: args.base_url.clone(),
            };
            let mut cfg = existing_config
                .clone()
                .unwrap_or_else(|| LashConfig::new(provider.clone()));
            cfg.upsert_provider(provider.clone());
            let _ = cfg.set_active_provider_kind(provider.kind());
            cfg.set_tavily_api_key(args.tavily_api_key.clone());
            cfg
        } else if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            // ANTHROPIC_API_KEY env var → activate direct Claude bearer token.
            let provider = Provider::Claude {
                access_token: key,
                refresh_token: String::new(),
                expires_at: u64::MAX,
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
        if c.active_provider_mut().ensure_fresh().await? {
            c.save()?; // persist refreshed tokens
        }
        c
    };

    // CLI env/flags override stored config
    if let Some(ref key) = args.tavily_api_key {
        lash_config.set_tavily_api_key(Some(key.clone()));
    }
    let context_folding = resolve_context_folding(
        lash_config.context_folding(),
        args.context_fold_soft_pct,
        args.context_fold_hard_pct,
    )
    .map_err(anyhow::Error::msg)?;
    lash_config.set_context_folding(context_folding);
    if args.print_prompt.is_none() {
        lash_config.save()?;
    }
    let model_catalog = models_dev_catalog().map_err(anyhow::Error::msg)?;
    if let Err(err) = model_catalog
        .refresh_if_stale(lash_core::model_info::DEFAULT_REFRESH_INTERVAL)
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
            let dir = lash_core::lash_home().join("sessions");
            Some(dir.join(format!(
                "{}.llm.jsonl",
                chrono::Local::now().format("%Y%m%d_%H%M%S")
            )))
        } else {
            None
        }
    });

    let sessions_dir = lash_core::lash_home().join("sessions");
    std::fs::create_dir_all(&sessions_dir)?;
    let resume_start = if let Some(filename) = args.resume.as_deref() {
        Some(
            session_log::load_session_start(filename)
                .ok_or_else(|| anyhow::anyhow!("Could not load session metadata: {}", filename))?,
        )
    } else {
        None
    };

    let autonomous = args.print_prompt.is_some();
    if autonomous {
        prompt_overrides.extend(autonomous_prompt_overrides());
    }
    let run_session_id = if autonomous {
        None
    } else {
        resume_start
            .as_ref()
            .map(|start| start.session_id.clone())
            .or_else(|| Some(uuid::Uuid::new_v4().to_string()))
    };
    let instruction_source: Arc<dyn InstructionSource> = Arc::new(FsInstructionSource::new());
    let mut capabilities = lash_core::AgentCapabilities::default();
    if autonomous {
        capabilities = capabilities.disable(lash_core::CapabilityId::Ask);
    }

    let config = AgentConfig {
        model: model.clone(),
        provider: lash_config.active_provider().clone(),
        model_variant,
        max_context_tokens: Some(resolved_model_spec.context_window() as usize),
        session_id: run_session_id.clone(),
        llm_log_path,
        prompt_overrides,
        execution_mode,
        context_folding,
        instruction_source: Arc::clone(&instruction_source),
        capabilities,
        ..Default::default()
    };

    // Build store (SQLite-backed archive)
    let db_path = if let Some(filename) = args.resume.as_deref() {
        sessions_dir.join(format!("{}.db", filename.trim_end_matches(".jsonl")))
    } else {
        sessions_dir.join(format!(
            "{}.db",
            chrono::Local::now().format("%Y%m%d_%H%M%S")
        ))
    };
    let store = Arc::new(Store::open(&db_path).expect("Failed to open session database"));

    let skill_dirs = vec![
        lash_core::lash_home().join("skills"),
        lash_core::legacy_repo_local_lash_dir().join("skills"),
        lash_core::repo_local_lash_dir().join("skills"),
    ];
    let tavily_key = lash_config.tavily_api_key().unwrap_or_default().to_string();
    let prompt_bridge = PromptBridge::new();
    let turn_injection_bridge = TurnInjectionBridge::new();
    let base_all = ToolSet::defaults_for(
        execution_mode,
        ToolSetDeps {
            store: Some(Arc::clone(&store)),
            tavily_api_key: if tavily_key.is_empty() {
                None
            } else {
                Some(tavily_key)
            },
            skill_dirs: Some(skill_dirs),
            prompt_bridge: (!autonomous).then_some(prompt_bridge.clone()),
        },
    );
    let base_provider: Arc<dyn ToolProvider> = Arc::new(base_all);
    let plugin_host = PluginHost::new(vec![
        Arc::new(BuiltinPromptContextPluginFactory::new(
            Arc::clone(&instruction_source),
            PromptContextPluginConfig::default(),
        )),
        Arc::new(BuiltinHistoryPluginFactory::new(Arc::clone(&store))),
        Arc::new(BuiltinMemoryPluginFactory::new(Arc::clone(&store))),
        Arc::new(BuiltinPlanTrackerPluginFactory),
        Arc::new(BuiltinPlanModePluginFactory::default()),
        Arc::new(BuiltinToolSurfacePluginFactory),
        Arc::new(fork::ForkPluginFactory),
    ]);
    let root_plugins = plugin_host.build_session("root", None)?;

    let mut all_base = ToolSet::new().with_arc_provider(Arc::clone(&base_provider));
    let mut tool_names: BTreeSet<String> = base_provider
        .definitions()
        .into_iter()
        .map(|def| def.name)
        .collect();
    for provider in root_plugins.tool_providers() {
        for def in provider.definitions() {
            if !tool_names.insert(def.name.clone()) {
                return Err(anyhow::anyhow!(format!(
                    "duplicate tool name registered by plugin runtime: {}",
                    def.name
                )));
            }
        }
        all_base = all_base.with_arc_provider(Arc::clone(provider));
    }
    let all_base_tools: Arc<dyn ToolProvider> = Arc::new(all_base);

    let agent_parent_tools: Arc<SwitchableTools> =
        Arc::new(SwitchableTools::new(Arc::clone(&all_base_tools)));

    // Root cancel token — lives for the whole app lifetime, child tokens are created per run
    let root_cancel = CancellationToken::new();
    let agent_call_config = AgentCallConfig {
        low_tier_execution_mode: lash_config
            .runtime
            .low_tier_subagent_execution_mode
            .unwrap_or(ExecutionMode::Standard),
    };
    let probe_agent_call = AgentCall::new(
        Arc::clone(&agent_parent_tools) as Arc<dyn ToolProvider>,
        Arc::clone(&root_plugins),
        &config,
        agent_call_config,
        lash_config.agent_models.clone(),
        root_cancel.clone(),
    );
    let mut capability_defs = default_dynamic_capability_defs();
    for (id, def) in root_plugins.capability_defs() {
        capability_defs.insert(id.clone(), def.clone());
    }
    let mut resolver_catalog = all_base_tools.definitions();
    resolver_catalog.extend(probe_agent_call.definitions());
    let resolved =
        resolve_capability_projection(&capability_defs, &config.capabilities, &resolver_catalog)?;

    let base_allowed: BTreeSet<String> = all_base_tools
        .definitions()
        .into_iter()
        .map(|d| d.name)
        .filter(|n| resolved.effective_tools.contains(n))
        .collect();
    let filtered_base: Arc<dyn ToolProvider> = Arc::new(FilteredTools::new(
        Arc::clone(&all_base_tools),
        base_allowed,
    ));
    agent_parent_tools.swap(Arc::clone(&filtered_base));

    let agent_allowed: BTreeSet<String> = probe_agent_call
        .definitions()
        .into_iter()
        .map(|d| d.name)
        .filter(|n| resolved.effective_tools.contains(n))
        .collect();

    let mut tools_comp = ToolSet::new().with_arc_provider(Arc::clone(&filtered_base));
    if !agent_allowed.is_empty() {
        let agent_call = AgentCall::new(
            Arc::clone(&agent_parent_tools) as Arc<dyn ToolProvider>,
            Arc::clone(&root_plugins),
            &config,
            agent_call_config,
            lash_config.agent_models.clone(),
            root_cancel.clone(),
        );
        let filtered_agent: Arc<dyn ToolProvider> =
            Arc::new(FilteredTools::new(Arc::new(agent_call), agent_allowed));
        tools_comp = tools_comp.with_arc_provider(filtered_agent);
    }
    let tools: Arc<dyn ToolProvider> = Arc::new(tools_comp);
    let dynamic_tools = Arc::new(DynamicToolProvider::from_tool_provider(
        Arc::clone(&tools),
        capability_defs,
        profile_from_agent_capabilities(&config.capabilities),
    )?);
    let dynamic_tools_provider: Arc<dyn ToolProvider> = dynamic_tools.clone();
    agent_parent_tools.swap(dynamic_tools_provider.clone());
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
                config.model_variant.as_deref(),
                execution_mode,
                Some(resolved_model_spec.context_window()),
                dynamic_tools_provider.definitions().len(),
                &toolset_hash,
                context_folding,
                &cwd,
                None,
            )
        );
        return Ok(());
    }
    let initial_model_variant = config.model_variant.clone();
    let runtime_config: RuntimeConfig = config.clone().into();
    let runtime = LashRuntime::from_state(
        runtime_config,
        RuntimeServices::new_with_bridges(
            dynamic_tools_provider,
            root_plugins,
            prompt_bridge,
            turn_injection_bridge.clone(),
        ),
        AgentStateEnvelope {
            agent_id: "root".to_string(),
            execution_mode,
            context_folding,
            ..AgentStateEnvelope::default()
        },
    )
    .await?;

    // ── Autonomous preset: skip TUI, run agent, print response to stdout ──
    if let Some(prompt) = args.print_prompt {
        return run_autonomous(runtime, prompt).await;
    }

    let session_name = resume_start
        .as_ref()
        .map(|start| start.session_name.clone())
        .unwrap_or_else(|| generate_session_name(&sessions_dir));
    let mut logger = if let Some(filename) = args.resume.as_deref() {
        SessionLogger::resume(filename)?
    } else {
        SessionLogger::new(&model, run_session_id, session_name.clone())?
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

/// Run the agent autonomously: send prompt, consume events, print final response to stdout.
async fn run_autonomous(mut runtime: LashRuntime, prompt: String) -> anyhow::Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};

    struct AutonomousSink {
        had_error: AtomicBool,
    }

    #[async_trait::async_trait]
    impl EventSink for AutonomousSink {
        async fn emit(&self, event: AgentEvent) {
            match event {
                AgentEvent::Error { message, .. } => {
                    eprintln!("error: {}", message);
                    self.had_error.store(true, Ordering::Relaxed);
                }
                AgentEvent::Prompt { response_tx, .. } => {
                    // No human is attached in the autonomous preset.
                    let _ = response_tx.send(String::new());
                }
                _ => {}
            }
        }
    }

    let (items, image_blobs) = build_items_from_editor_input(&prompt, Vec::new());
    let sink = AutonomousSink {
        had_error: AtomicBool::new(false),
    };
    let result = runtime
        .stream_turn(
            TurnInput {
                items,
                image_blobs,
                mode: Some(RunMode::Normal),
            },
            &sink,
            CancellationToken::new(),
        )
        .await;

    match result {
        Ok(turn) => {
            if !turn.assistant_output.safe_text.is_empty() {
                println!("{}", turn.assistant_output.safe_text);
            } else {
                let raw = turn.assistant_output.raw_text.trim();
                if raw.is_empty() {
                    eprintln!("error: model returned no usable assistant output");
                } else {
                    let mut preview: String = raw.chars().take(64).collect();
                    if raw.chars().count() > 64 {
                        preview.push_str("...");
                    }
                    eprintln!(
                        "error: model returned malformed assistant output: {}",
                        preview
                    );
                }
                std::process::exit(2);
            }
        }
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    }
    if sink.had_error.load(Ordering::Relaxed) {
        std::process::exit(1);
    }
    Ok(())
}

/// Returned by the spawned runtime task so we can reclaim ownership.
struct RuntimeRunResult {
    stream_id: u64,
    runtime: LashRuntime,
    result: AssembledTurn,
}

#[derive(Clone)]
struct TurnReplayPayload {
    display_input: String,
    turn_input: TurnInput,
    execution_mode: ExecutionMode,
}

struct AppEventSink {
    tx: mpsc::UnboundedSender<AppEvent>,
    stream_id: u64,
}

#[async_trait::async_trait]
impl EventSink for AppEventSink {
    async fn emit(&self, event: AgentEvent) {
        let _ = self.tx.send(AppEvent::Agent {
            stream_id: self.stream_id,
            event,
        });
    }
}

/// Build the controls text shown by /controls.
fn controls_text() -> String {
    [
        "Controls:",
        "  Esc                Cancel agent (while running)",
        "  Enter              Submit; inject at next checkpoint while running",
        "  Tab                Queue next turn; submit plain draft when idle",
        "  Alt+Up             Edit last queued turn",
        "  Shift+Tab          Toggle persistent plan mode",
        "  Ctrl+U / Ctrl+D    Scroll half-page up / down",
        "  PgUp / PgDn        Scroll page up / down",
        "  Shift+Enter        Insert newline",
        "  Ctrl+V             Paste image as inline [Image #n]",
        "  Ctrl+Shift+V       Paste text only",
        "  Ctrl+Y             Copy last response to clipboard",
        "  Ctrl+O             Cycle tool expansion (ghost ↔ compact)",
        "  Alt+O              Full expansion (code + stdout)",
        "  Up / Down          Input history",
        "  Shift+Drag         Select text (terminal native)",
        "  Ctrl+C             Quit",
    ]
    .join("\n")
}

#[derive(Debug, Clone)]
struct ModelSelection {
    model: String,
}

fn models_dev_catalog() -> Result<Arc<CachedModelCatalog>, String> {
    CachedModelCatalog::models_dev(
        Arc::new(FileModelCatalogStore::default_models_dev()),
        Some(Arc::new(ModelsDevHttpSource::default_models_dev())),
    )
    .map(Arc::new)
}

fn parse_model_selection(input: &str) -> Result<ModelSelection, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("Model cannot be empty.".to_string());
    }

    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    match parts.as_slice() {
        [model] => Ok(ModelSelection {
            model: (*model).to_string(),
        }),
        _ => Err("Model input must be a single `<model>` token.".to_string()),
    }
}

fn parse_variant_input(input: &str) -> Result<String, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("Variant cannot be empty.".to_string());
    }
    if trimmed.contains(char::is_whitespace) {
        return Err("Variant must be a single token.".to_string());
    }
    Ok(trimmed.to_ascii_lowercase())
}

fn parse_execution_mode(input: &str) -> Result<ExecutionMode, String> {
    match input.trim().to_ascii_lowercase().as_str() {
        "" => Err("Execution mode cannot be empty.".to_string()),
        "repl" => Ok(ExecutionMode::Repl),
        "standard" | "tools" => Ok(ExecutionMode::Standard),
        other => Err(format!(
            "Unknown execution mode `{other}`. Expected `repl` or `standard`."
        )),
    }
}

fn execution_mode_usage() -> &'static str {
    if lash_core::execution_mode_supported(ExecutionMode::Repl) {
        "<repl|standard>"
    } else {
        "<standard>"
    }
}

fn ensure_supported_execution_mode(mode: ExecutionMode) -> Result<ExecutionMode, String> {
    if lash_core::execution_mode_supported(mode) {
        Ok(mode)
    } else {
        Err(match mode {
            ExecutionMode::Repl => "REPL mode is not available in this build.".to_string(),
            ExecutionMode::Standard => "Execution mode is not available.".to_string(),
        })
    }
}

fn execution_mode_label(mode: ExecutionMode) -> &'static str {
    match mode {
        ExecutionMode::Repl => "repl",
        ExecutionMode::Standard => "standard",
    }
}

fn resolve_context_folding(
    configured: ContextFoldingConfig,
    soft_override: Option<u8>,
    hard_override: Option<u8>,
) -> Result<ContextFoldingConfig, String> {
    ContextFoldingConfig {
        soft_limit_pct: soft_override.unwrap_or(configured.soft_limit_pct),
        hard_limit_pct: hard_override.unwrap_or(configured.hard_limit_pct),
    }
    .validate()
}

fn validate_model_selection(provider: &Provider, selection: &ModelSelection) -> Result<(), String> {
    provider
        .validate_model_name(&selection.model)
        .map_err(|err| {
            let normalized = provider.resolve_model(&selection.model);
            if normalized != selection.model {
                format!("{err}\nResolved provider model ID: `{normalized}`")
            } else {
                err
            }
        })
}

fn resolve_model_selection(
    provider: &Provider,
    selection: &ModelSelection,
    catalog: &CachedModelCatalog,
) -> Result<ResolvedModelSpec, String> {
    let snapshot = catalog.snapshot();
    provider
        .resolve_model_spec(&selection.model, &snapshot)
        .map_err(|err| {
            let normalized = provider.resolve_model(&selection.model);
            if normalized != selection.model {
                format!("{err}\nResolved provider model ID: `{normalized}`")
            } else {
                err
            }
        })
}

fn resolve_model_variant(
    provider: &Provider,
    model: &str,
    requested: Option<&str>,
) -> Result<Option<String>, String> {
    let Some(raw) = requested else {
        return Ok(provider.default_model_variant(model).map(str::to_string));
    };
    let variant = parse_variant_input(raw)?;
    if variant == "default" {
        return Ok(provider.default_model_variant(model).map(str::to_string));
    }
    provider.validate_variant(model, &variant)?;
    Ok(Some(variant))
}

fn variant_lines(provider: &Provider, model: &str, current_variant: Option<&str>) -> Vec<String> {
    let supported = provider.supported_variants(model);
    let mut lines = Vec::new();
    if supported.is_empty() {
        lines.push(format!(
            "`{}` on {} does not expose configurable variants.",
            model,
            provider.label()
        ));
        return lines;
    }
    lines.push(format!(
        "Current variant: `{}`",
        current_variant.unwrap_or("(none)")
    ));
    if let Some(default_variant) = provider.default_model_variant(model) {
        lines.push(format!("Recommended default: `{}`", default_variant));
    }
    lines.push(format!("Available variants: {}", supported.join(", ")));
    lines.push("Usage: `/variant <name>` or `/variant default`".to_string());
    lines
}

fn hash12(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{:x}", digest)[..12].to_string()
}

fn latest_user_prompt_hash(messages: &[Message]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == MessageRole::User)
        .map(|m| {
            let text = m
                .parts
                .iter()
                .map(|p| p.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            hash12(text.as_bytes())
        })
}

struct ReplayManifest {
    version: u8,
    saved_at: String,
    provider: String,
    configured_model: String,
    resolved_model: String,
    context_window: u64,
    model_variant: Option<String>,
    toolset_hash: String,
    prompt_hash: Option<String>,
    snapshot_hash: Option<String>,
}

#[allow(clippy::too_many_arguments)]
fn persist_root_agent_state(
    store: &Store,
    state: &mut AgentStateEnvelope,
    dynamic_state: &DynamicStateSnapshot,
    provider: &Provider,
    configured_model: &str,
    context_window: u64,
    execution_mode: ExecutionMode,
    context_folding: ContextFoldingConfig,
    model_variant: Option<&str>,
    toolset_hash: &str,
    prompt_hash: Option<String>,
    snapshot_hash: Option<String>,
) {
    let manifest = ReplayManifest {
        version: 3,
        saved_at: chrono::Utc::now().to_rfc3339(),
        provider: provider.id().to_string(),
        configured_model: configured_model.to_string(),
        resolved_model: provider.resolve_model(configured_model),
        context_window,
        model_variant: model_variant.map(str::to_string),
        toolset_hash: toolset_hash.to_string(),
        prompt_hash,
        snapshot_hash,
    };
    let manifest_json = serde_json::json!({
        "version": manifest.version,
        "saved_at": manifest.saved_at,
        "provider": manifest.provider,
        "configured_model": manifest.configured_model,
        "resolved_model": manifest.resolved_model,
        "context_window": manifest.context_window,
        "execution_mode": execution_mode_label(execution_mode),
        "context_folding": {
            "soft_limit_pct": context_folding.soft_limit_pct,
            "hard_limit_pct": context_folding.hard_limit_pct,
        },
        "model_variant": manifest.model_variant,
        "toolset_hash": manifest.toolset_hash,
        "prompt_hash": manifest.prompt_hash,
        "snapshot_hash": manifest.snapshot_hash,
    });
    state.replay_manifest = Some(manifest_json.clone());
    let config_json = serde_json::json!({
        "manifest": manifest_json,
        "context_folding": {
            "soft_limit_pct": context_folding.soft_limit_pct,
            "hard_limit_pct": context_folding.hard_limit_pct,
        },
        "plugin_snapshot": state.plugin_snapshot,
        "task_state": state.task_state,
        "subagent_state": state.subagent_state,
        "dynamic_state": dynamic_state,
    })
    .to_string();
    let messages_json = serde_json::to_string(&state.messages).unwrap_or_else(|_| "[]".to_string());
    store.save_agent_state(
        "root",
        None,
        "active",
        &messages_json,
        state.iteration as i64,
        &config_json,
        state.repl_snapshot.as_deref(),
        state.token_usage.input_tokens,
        state.token_usage.output_tokens,
        state.token_usage.cached_input_tokens,
    );
}

fn push_system_message(app: &mut App, msg: impl Into<String>) {
    let msg = msg.into();
    let duplicate = matches!(
        app.blocks.last(),
        Some(DisplayBlock::SystemMessage(existing)) if existing == &msg
    );
    if duplicate {
        return;
    }
    app.blocks.push(DisplayBlock::SystemMessage(msg));
    app.invalidate_height_cache();
    app.scroll_to_bottom();
}

fn version_text() -> String {
    format!("lash {}\nlash-core {}", APP_VERSION, lash_core::VERSION)
}

fn info_text_unconfigured(execution_mode: ExecutionMode, cwd: &str) -> String {
    [
        format!("lash: {}", APP_VERSION),
        format!("lash-core: {}", lash_core::VERSION),
        "provider: (not configured)".to_string(),
        "configured model: (not configured)".to_string(),
        "resolved model: (not configured)".to_string(),
        format!("execution mode: {}", execution_mode_label(execution_mode)),
        "context window: unknown".to_string(),
        format!("cwd: {}", cwd),
        "session: (not started)".to_string(),
    ]
    .join("\n")
}

#[allow(clippy::too_many_arguments)]
fn info_text(
    provider: &Provider,
    configured_model: &str,
    model_variant: Option<&str>,
    execution_mode: ExecutionMode,
    context_window: Option<u64>,
    tool_count: usize,
    toolset_hash: &str,
    context_folding: ContextFoldingConfig,
    cwd: &str,
    session_name: Option<&str>,
) -> String {
    let resolved_model = provider.resolve_model(configured_model);
    let mut lines = vec![
        format!("lash: {}", APP_VERSION),
        format!("lash-core: {}", lash_core::VERSION),
        format!("provider: {} ({})", provider.label(), provider.id()),
        format!("configured model: {}", configured_model),
        format!("resolved model: {}", resolved_model),
        format!("execution mode: {}", execution_mode_label(execution_mode)),
    ];

    if let Some(variant) = model_variant {
        lines.push(format!("variant: {}", variant));
    }
    if let Some(window) = context_window {
        lines.push(format!("context window: {}", window));
    } else {
        lines.push("context window: unknown".to_string());
    }

    lines.extend([
        format!(
            "context folding: soft={}%, hard={}%",
            context_folding.soft_limit_pct, context_folding.hard_limit_pct
        ),
        format!("tools: {} (hash {})", tool_count, toolset_hash),
        format!("cwd: {}", cwd),
        format!("session: {}", session_name.unwrap_or("(not started)")),
    ]);

    lines.join("\n")
}

/// Build the help text shown by /help.
fn help_text(skills: &skill::SkillRegistry) -> String {
    let mut lines = vec![
        "Commands:".to_string(),
        "  /clear, /new       Reset conversation".to_string(),
        "  /fork [prompt]     Open a forked session in a new terminal".to_string(),
        "  /version           Show Lash and lash-core versions".to_string(),
        "  /info              Show current runtime/session info".to_string(),
        "  /model [name]      Show or switch LLM model".to_string(),
        "  /variant [name]    Show or switch provider-native model variant".to_string(),
        format!(
            "  /mode [name]       Show current execution mode; new session required to change {}",
            execution_mode_usage()
        ),
        "  /provider          Switch, add, or re-authenticate providers".to_string(),
        "  /login             Sign in or reconfigure provider".to_string(),
        "  /logout            Remove stored credentials for active provider".to_string(),
        "  /retry             Replay the previous turn payload".to_string(),
        "  /resume [name]     Browse or load a previous session".to_string(),
        "  /skills            Browse loaded skills".to_string(),
        "  /tools ...         Inspect/edit dynamic tools".to_string(),
        "  /caps ...          Inspect/edit dynamic capabilities".to_string(),
        "  /reconfigure ...   Apply/status/clear pending changes".to_string(),
        "  /help, /?          Show this help".to_string(),
        "  /exit, /quit       Quit".to_string(),
    ];

    if !skills.is_empty() {
        lines.push(String::new());
        lines.push("Skills:".to_string());
        for skill in skills.iter() {
            let desc = if skill.description.is_empty() {
                String::new()
            } else {
                format!("  {}", skill.description)
            };
            lines.push(format!("  /{}{}", skill.name, desc));
        }
    }

    lines.extend([
        String::new(),
        "Dynamic Runtime:".to_string(),
        "  /tools".to_string(),
        "  /tools add <name> <handler> [description]".to_string(),
        "  /tools rm <name>".to_string(),
        "  /tools update <name> key=value ...".to_string(),
        "  /caps".to_string(),
        "  /caps add <id> name=<n> tools=a,b helpers=h1,h2 prompt=<text>".to_string(),
        "  /caps rm <id>".to_string(),
        "  /caps enable <id> | /caps disable <id>".to_string(),
        "  /caps tool-enable <tool> | /caps tool-disable <tool>".to_string(),
        "  /reconfigure status|apply|clear".to_string(),
    ]);

    lines.extend([
        String::new(),
        "Shortcuts:".to_string(),
        "  Esc                Cancel agent (while running)".to_string(),
        "  Enter              Submit; inject at next checkpoint while running".to_string(),
        "  Tab                Queue next turn; submit plain draft when idle".to_string(),
        "  Alt+Up             Edit last queued turn".to_string(),
        "  Shift+Tab          Toggle persistent plan mode".to_string(),
        "  Ctrl+U / Ctrl+D    Scroll half-page up / down".to_string(),
        "  PgUp / PgDn        Scroll page up / down".to_string(),
        "  Shift+Enter        Insert newline".to_string(),
        "  Ctrl+V             Paste image as inline [Image #n]".to_string(),
        "  Ctrl+Shift+V       Paste text only".to_string(),
        "  Ctrl+Y             Copy last response to clipboard".to_string(),
        "  Ctrl+O             Cycle tool expansion (ghost \u{2194} compact)".to_string(),
        "  Alt+O              Full expansion (code + stdout)".to_string(),
        "  Shift+Drag         Select text (terminal native)".to_string(),
        "  Up/Down            Input history".to_string(),
        "  Ctrl+C             Quit".to_string(),
    ]);

    lines.join("\n")
}

fn plan_mode_enabled_from_result(result: ToolResult) -> Result<bool, String> {
    if !result.success {
        return Err(result.result.to_string());
    }
    result
        .result
        .get("enabled")
        .and_then(|value| value.as_bool())
        .ok_or_else(|| "plan mode API response missing `enabled`".to_string())
}

async fn plan_mode_status(
    plugin_host: &PluginHost,
    session_manager: Arc<dyn SessionManager>,
) -> Result<bool, String> {
    let result = plugin_host
        .invoke_external_for_session(
            ROOT_SESSION_ID,
            "plan_mode.status",
            serde_json::json!({}),
            session_manager,
        )
        .await
        .map_err(|err| err.to_string())?;
    plan_mode_enabled_from_result(result)
}

async fn plan_mode_toggle(
    plugin_host: &PluginHost,
    session_manager: Arc<dyn SessionManager>,
) -> Result<bool, String> {
    let result = plugin_host
        .invoke_external_for_session(
            ROOT_SESSION_ID,
            "plan_mode.toggle",
            serde_json::json!({}),
            session_manager,
        )
        .await
        .map_err(|err| err.to_string())?;
    plan_mode_enabled_from_result(result)
}

async fn sync_plan_mode(
    app: &mut App,
    plugin_host: &PluginHost,
    session_manager: Arc<dyn SessionManager>,
) {
    match plan_mode_status(plugin_host, session_manager).await {
        Ok(enabled) => app.set_plan_mode_enabled(enabled),
        Err(err) => push_system_message(app, format!("Failed to read plan mode: {}", err)),
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_app(
    mut terminal: DefaultTerminal,
    runtime: LashRuntime,
    plugin_host: PluginHost,
    dynamic_tools: Arc<DynamicToolProvider>,
    turn_injection_bridge: TurnInjectionBridge,
    logger: &mut SessionLogger,
    args: &Args,
    mut provider: Provider,
    model: String,
    initial_context_window: u64,
    session_name: String,
    model_catalog: Arc<CachedModelCatalog>,
    store: Arc<Store>,
    mut toolset_hash: String,
    initial_model_variant: Option<String>,
    initial_execution_mode: ExecutionMode,
) -> anyhow::Result<()> {
    let mut app = App::new(model, session_name);
    app.context_window = Some(initial_context_window);
    app.context_usage_excludes_cached_input = provider.input_usage_excludes_cached_tokens();
    let mut current_model_variant = initial_model_variant.or_else(|| {
        provider
            .default_model_variant(&app.model)
            .map(str::to_string)
    });
    app.set_model_variant(current_model_variant.clone());
    let mut current_execution_mode = initial_execution_mode;
    app.load_history();
    let mut history: Vec<Message> = Vec::new();
    let mut turn_counter: usize = 0;
    let mut session_manager = runtime
        .session_manager()
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let mut runtime = Some(runtime);
    let mut current_context_folding = runtime
        .as_ref()
        .map(|rt| rt.export_state().context_folding)
        .unwrap_or_default();
    let mut desired_dynamic = dynamic_tools.export_state();
    let mut pending_reconfigure = false;

    // Cancellation token for interrupting a running agent
    let mut cancel_token: Option<CancellationToken> = None;

    // Unified event channel
    let (app_tx, mut app_rx) = mpsc::unbounded_channel::<AppEvent>();

    // Stop/pause flags for terminal event pumps.
    let stop = Arc::new(AtomicBool::new(false));
    let paused = Arc::new(AtomicBool::new(false));

    // Spawn terminal event reader using poll() with timeout so it can stop
    let term_tx = app_tx.clone();
    let stop_reader = Arc::clone(&stop);
    let paused_reader = Arc::clone(&paused);
    tokio::task::spawn_blocking(move || {
        while !stop_reader.load(Ordering::Relaxed) {
            if paused_reader.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(50));
                continue;
            }
            // Poll with 50ms timeout so we can check the stop flag
            if crossterm::event::poll(std::time::Duration::from_millis(50)).unwrap_or(false) {
                match crossterm::event::read() {
                    Ok(ev) => {
                        if term_tx.send(AppEvent::Terminal(ev)).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        }
    });

    // Tick timer for spinner animation
    let tick_tx = app_tx.clone();
    let stop_tick = Arc::clone(&stop);
    let paused_tick = Arc::clone(&paused);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(100));
        loop {
            interval.tick().await;
            if stop_tick.load(Ordering::Relaxed) {
                break;
            }
            if paused_tick.load(Ordering::Relaxed) {
                continue;
            }
            if tick_tx.send(AppEvent::Tick).is_err() {
                break;
            }
        }
    });

    #[cfg(unix)]
    {
        // SIGTERM handler for graceful shutdown
        let sigterm_tx = app_tx.clone();
        tokio::spawn(async move {
            use tokio::signal::unix::{SignalKind, signal};
            if let Ok(mut sig) = signal(SignalKind::terminate()) {
                sig.recv().await;
                let _ = sigterm_tx.send(AppEvent::Quit);
            }
        });
    }

    // Oneshot for receiving runtime back after a run completes
    let mut runtime_return_rx: Option<tokio::sync::oneshot::Receiver<RuntimeRunResult>> = None;
    let mut last_turn: Option<TurnReplayPayload> = None;
    let mut active_stream_id: u64 = 0;
    let mut pending_clear_after_return = false;

    sync_plan_mode(&mut app, &plugin_host, Arc::clone(&session_manager)).await;

    if let Some(filename) = args.resume.as_deref() {
        if let Err(err) = resume::load_resumed_session(
            filename,
            &mut app,
            &mut history,
            &mut runtime,
            &mut turn_counter,
            &mut current_execution_mode,
            &provider,
            &mut current_model_variant,
            &dynamic_tools,
            &mut desired_dynamic,
            model_catalog.as_ref(),
        )
        .await
        {
            push_system_message(&mut app, err);
        } else {
            if let Some(rt) = runtime.as_ref() {
                match rt.session_manager() {
                    Ok(manager) => session_manager = manager,
                    Err(err) => push_system_message(
                        &mut app,
                        format!("Failed to refresh session manager: {}", err),
                    ),
                }
            }
            sync_plan_mode(&mut app, &plugin_host, Arc::clone(&session_manager)).await;
            toolset_hash = hash12(
                &serde_json::to_vec(&dynamic_tools.definitions())
                    .unwrap_or_else(|_| b"[]".to_vec()),
            );
        }
        if let Some(prompt) = args
            .resume_prompt
            .as_deref()
            .map(str::trim)
            .filter(|prompt| !prompt.is_empty())
        {
            if let Err(e) = apply_pending_reconfigure(
                &dynamic_tools,
                &mut desired_dynamic,
                &mut pending_reconfigure,
                &mut runtime,
            )
            .await
            {
                push_system_message(
                    &mut app,
                    format!(
                        "Pending runtime reconfigure failed; startup prompt blocked: {}",
                        e
                    ),
                );
            } else {
                toolset_hash = hash12(
                    &serde_json::to_vec(&dynamic_tools.definitions())
                        .unwrap_or_else(|_| b"[]".to_vec()),
                );
                let (items, image_blobs) = build_items_from_editor_input(prompt, Vec::new());
                let turn_input = make_turn_input(&mut app, items, image_blobs);
                send_user_message(
                    prompt.to_string(),
                    turn_input.clone(),
                    &mut app,
                    logger,
                    &mut runtime,
                    &mut history,
                    &mut runtime_return_rx,
                    &mut cancel_token,
                    &mut active_stream_id,
                    &app_tx,
                );
                last_turn = Some(TurnReplayPayload {
                    display_input: prompt.to_string(),
                    turn_input,
                    execution_mode: current_execution_mode,
                });
            }
        }
    }

    loop {
        // Check if runtime turn completed — reclaim runtime + updated history
        if let Some(ref mut rx) = runtime_return_rx {
            match rx.try_recv() {
                Ok(done) => {
                    runtime = Some(done.runtime);
                    if done.stream_id != active_stream_id || pending_clear_after_return {
                        if let Some(rt) = runtime.as_mut() {
                            let _ = rt.reset_session().await;
                            rt.set_state(AgentStateEnvelope::default());
                        }
                        history.clear();
                        turn_counter = 0;
                        app.token_usage = TokenUsage::default();
                        app.running = false;
                        app.set_plan_mode_enabled(false);
                        app.set_model_variant(current_model_variant.clone());
                        runtime_return_rx = None;
                        cancel_token = None;
                        pending_clear_after_return = false;
                        app.dirty = true;
                        continue;
                    }
                    let mut state = done.result.state;
                    tracing::info!(
                        iteration = state.iteration,
                        status = ?done.result.status,
                        reason = ?done.result.done_reason,
                        assistant_chars = done.result.assistant_output.safe_text.len(),
                        "runtime turn completed"
                    );
                    let no_visible_output = matches!(done.result.status, TurnStatus::Completed)
                        && done.result.assistant_output.safe_text.trim().is_empty()
                        && done.result.code_outputs.is_empty()
                        && done.result.errors.is_empty();
                    if no_visible_output {
                        let raw = done.result.assistant_output.raw_text.trim();
                        if raw.is_empty() {
                            push_system_message(
                                &mut app,
                                "Model returned no usable output. Use `/retry` to replay the last turn.",
                            );
                        } else {
                            let mut preview: String = raw.chars().take(48).collect();
                            if raw.chars().count() > 48 {
                                preview.push_str("...");
                            }
                            let preview = preview.replace('`', "'");
                            push_system_message(
                                &mut app,
                                format!(
                                    "Model returned malformed output (`{}`). Use `/retry` to replay the last turn.",
                                    preview
                                ),
                            );
                        }
                    }

                    // Snapshot REPL after each completed turn so resume can restore exact state.
                    let snapshot_hash = if let Some(rt) = runtime.as_mut() {
                        if matches!(state.execution_mode, ExecutionMode::Repl) {
                            match rt.snapshot_repl().await {
                                Ok(blob) => {
                                    state = rt.export_state();
                                    Some(hash12(&blob))
                                }
                                Err(e) => {
                                    push_system_message(
                                        &mut app,
                                        format!(
                                            "Warning: failed to snapshot REPL state for resume: {}",
                                            e
                                        ),
                                    );
                                    None
                                }
                            }
                        } else {
                            state = rt.export_state();
                            None
                        }
                    } else {
                        None
                    };

                    history = state.messages.clone();
                    turn_counter = state.iteration;
                    app.token_usage = state.token_usage.clone();
                    current_context_folding = state.context_folding;

                    let persisted_execution_mode = state.execution_mode;
                    let persisted_context_folding = state.context_folding;
                    let persisted_dynamic_state = dynamic_tools.export_state();
                    persist_root_agent_state(
                        &store,
                        &mut state,
                        &persisted_dynamic_state,
                        &provider,
                        &app.model,
                        app.context_window
                            .expect("app context_window must be set before persisting state"),
                        persisted_execution_mode,
                        persisted_context_folding,
                        current_model_variant.as_deref(),
                        &toolset_hash,
                        latest_user_prompt_hash(&history),
                        snapshot_hash,
                    );
                    if let Some(rt) = runtime.as_mut() {
                        rt.set_state(state.clone());
                    }
                    runtime_return_rx = None;
                    cancel_token = None;
                    let leftover_injections =
                        turn_injection_bridge.drain().unwrap_or_else(|_| Vec::new());
                    if !leftover_injections.is_empty() {
                        while let Some(turn) = app.pending_steers.pop_front() {
                            app.queue_turn(turn);
                        }
                    }

                    if let Some((queued, was_pending)) = app.take_next_queued_turn() {
                        if let Err(e) = apply_pending_reconfigure(
                            &dynamic_tools,
                            &mut desired_dynamic,
                            &mut pending_reconfigure,
                            &mut runtime,
                        )
                        .await
                        {
                            push_system_message(
                                &mut app,
                                format!(
                                    "Pending runtime reconfigure failed; queued message not sent: {}",
                                    e
                                ),
                            );
                            app.requeue_front(queued, was_pending);
                            continue;
                        }
                        toolset_hash = hash12(
                            &serde_json::to_vec(&dynamic_tools.definitions())
                                .unwrap_or_else(|_| b"[]".to_vec()),
                        );
                        let (items, image_blobs) =
                            build_items_from_editor_input(&queued.text, queued.images.clone());
                        let turn_input = make_turn_input(&mut app, items, image_blobs);
                        send_user_message(
                            queued.text.clone(),
                            turn_input.clone(),
                            &mut app,
                            logger,
                            &mut runtime,
                            &mut history,
                            &mut runtime_return_rx,
                            &mut cancel_token,
                            &mut active_stream_id,
                            &app_tx,
                        );
                        last_turn = Some(TurnReplayPayload {
                            display_input: queued.text,
                            turn_input,
                            execution_mode: current_execution_mode,
                        });
                    }
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    app.running = false;
                    runtime_return_rx = None;
                    cancel_token = None;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {}
            }
        }

        // Draw only when dirty
        if app.dirty {
            // Pre-compute height cache before immutable borrow in draw
            let size = terminal.size()?;
            let vh = ui::history_viewport_height(&app, size.width, size.height);
            let vw = size.width as usize;
            app.ensure_height_cache_pub(vw, vh);
            app.refresh_follow_output_anchor(vw, vh);
            // Clamp scroll_offset (especially for scroll_to_bottom's usize::MAX)
            let total = app.total_content_height(vw, vh);
            let max_scroll = total.saturating_sub(vh);
            app.scroll_offset = app.scroll_offset.min(max_scroll);

            terminal.draw(|frame| ui::draw(frame, &app))?;
            app.dirty = false;
        }

        // Wait for next event
        let event = match app_rx.recv().await {
            Some(e) => e,
            None => break,
        };

        match event {
            AppEvent::Terminal(TermEvent::Key(key)) => {
                // With kitty keyboard protocol, ignore Release/Repeat events
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                // Re-enable mouse capture if it was released for native selection.
                if !app.mouse_captured && !args.no_mouse {
                    let _ = crossterm::execute!(
                        std::io::stdout(),
                        crossterm::event::EnableMouseCapture
                    );
                    app.mouse_captured = true;
                }
                app.dirty = true;
                // CTRL+C: dismiss prompt if active, else quit
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                    if app.has_prompt() {
                        app.dismiss_prompt();
                        continue;
                    }
                    break;
                }

                // ALT+O: reliable full expand toggle across most terminals.
                if key.modifiers.contains(KeyModifiers::ALT)
                    && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'o'))
                {
                    app.toggle_full_expand();
                    continue;
                }

                if key.modifiers.contains(KeyModifiers::ALT) && key.code == KeyCode::Up {
                    if let Some((turn, _was_pending)) = app.take_last_queued_turn() {
                        app.input = turn.text;
                        app.cursor_pos = app.input.len();
                        app.pending_images = turn.images;
                        app.update_suggestions();
                    }
                    continue;
                }

                // CTRL+O: cycle expand (0↔1)
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'o'))
                {
                    app.cycle_expand();
                    continue;
                }

                // CTRL+Y: copy last response to clipboard
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('y') {
                    copy_last_response(&app);
                    continue;
                }

                // CTRL+SHIFT+V: always paste text from clipboard
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && key.modifiers.contains(KeyModifiers::SHIFT)
                    && key.code == KeyCode::Char('V')
                {
                    if let Ok(mut clipboard) = arboard::Clipboard::new()
                        && let Ok(text) = clipboard.get_text()
                    {
                        for c in text.chars() {
                            app.insert_char(c);
                        }
                        app.update_suggestions();
                    }
                    continue;
                }

                // CTRL+V: paste image from clipboard (no text fallback)
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && key.code == KeyCode::Char('v')
                    && !app.running
                {
                    if let Ok(mut clipboard) = arboard::Clipboard::new() {
                        let got_image = clipboard.get_image().ok().and_then(|img_data| {
                            let w = img_data.width as u32;
                            let h = img_data.height as u32;
                            let rgba =
                                image::RgbaImage::from_raw(w, h, img_data.bytes.into_owned())?;
                            let mut png_buf = std::io::Cursor::new(Vec::new());
                            rgba.write_to(&mut png_buf, image::ImageFormat::Png).ok()?;
                            Some(png_buf.into_inner())
                        });
                        if let Some(png_bytes) = got_image {
                            app.pending_images.push(png_bytes);
                            let marker = format!("[Image #{}]", app.pending_images.len());
                            insert_inline_marker(&mut app, &marker);
                            app.update_suggestions();
                        }
                    }
                    continue;
                }

                // Escape key behavior depends on state
                if key.code == KeyCode::Esc {
                    if app.has_prompt() {
                        app.dismiss_prompt();
                    } else if app.has_skill_picker() {
                        app.dismiss_skill_picker();
                    } else if app.has_session_picker() {
                        app.dismiss_session_picker();
                    } else if app.running {
                        // Interrupt running agent
                        if let Some(token) = cancel_token.take() {
                            token.cancel();
                        }
                    }
                    // When idle with no dialog: no-op
                    continue;
                }

                // ── Always-on scroll keys (work in all states) ──
                {
                    let size = terminal.size()?;
                    let vh = ui::history_viewport_height(&app, size.width, size.height);
                    let vw = size.width as usize;
                    let half_page = vh / 2;

                    // Ctrl+U / Ctrl+D: half-page scroll
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('u')
                    {
                        app.scroll_up(half_page);
                        continue;
                    }
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('d')
                    {
                        app.scroll_down(half_page, vh, vw);
                        continue;
                    }

                    // PgUp / PgDn
                    if key.code == KeyCode::PageUp {
                        app.scroll_up(vh);
                        continue;
                    }
                    if key.code == KeyCode::PageDown {
                        app.scroll_down(vh, vh, vw);
                        continue;
                    }
                }

                // ── Skill picker key handling ──
                if app.has_skill_picker() {
                    match key.code {
                        KeyCode::Up | KeyCode::Char('k') => app.skill_picker_up(),
                        KeyCode::Down | KeyCode::Char('j') => app.skill_picker_down(),
                        KeyCode::Enter => {
                            if let Some(name) = app.take_skill_pick() {
                                app.input = format!("/{} ", name);
                                app.cursor_pos = app.input.len();
                            }
                        }
                        _ => {}
                    }
                    continue;
                }

                // ── Session picker key handling ──
                if app.has_session_picker() {
                    match key.code {
                        KeyCode::Up | KeyCode::Char('k') => app.session_picker_up(),
                        KeyCode::Down | KeyCode::Char('j') => app.session_picker_down(),
                        KeyCode::Enter => {
                            if let Some(filename) = app.take_session_pick() {
                                match resume::load_resumed_session(
                                    &filename,
                                    &mut app,
                                    &mut history,
                                    &mut runtime,
                                    &mut turn_counter,
                                    &mut current_execution_mode,
                                    &provider,
                                    &mut current_model_variant,
                                    &dynamic_tools,
                                    &mut desired_dynamic,
                                    model_catalog.as_ref(),
                                )
                                .await
                                {
                                    Ok(()) => {
                                        if let Some(rt) = runtime.as_ref() {
                                            match rt.session_manager() {
                                                Ok(manager) => session_manager = manager,
                                                Err(err) => push_system_message(
                                                    &mut app,
                                                    format!(
                                                        "Failed to refresh session manager: {}",
                                                        err
                                                    ),
                                                ),
                                            }
                                        }
                                        sync_plan_mode(
                                            &mut app,
                                            &plugin_host,
                                            Arc::clone(&session_manager),
                                        )
                                        .await;
                                        toolset_hash = hash12(
                                            &serde_json::to_vec(&dynamic_tools.definitions())
                                                .unwrap_or_else(|_| b"[]".to_vec()),
                                        );
                                    }
                                    Err(err) => {
                                        app.blocks.push(DisplayBlock::SystemMessage(err));
                                        app.invalidate_height_cache();
                                        app.scroll_to_bottom();
                                    }
                                }
                            }
                        }
                        _ => {} // ignore other keys while picker is open
                    }
                    continue;
                }

                // ── Prompt (ask dialog) key handling ──
                if app.has_prompt() {
                    match key.code {
                        KeyCode::Up => app.prompt_up(),
                        KeyCode::Down => app.prompt_down(),
                        KeyCode::BackTab => app.prompt_toggle_extra(),
                        KeyCode::Enter => {
                            app.take_prompt_response();
                        }
                        KeyCode::Char(c)
                            if app.is_prompt_editing_extra() || app.is_prompt_freeform() =>
                        {
                            app.prompt_insert_char(c);
                        }
                        KeyCode::Backspace
                            if app.is_prompt_editing_extra() || app.is_prompt_freeform() =>
                        {
                            app.prompt_backspace();
                        }
                        _ => {}
                    }
                    continue;
                }

                match key.code {
                    KeyCode::BackTab => {
                        match plan_mode_toggle(&plugin_host, Arc::clone(&session_manager)).await {
                            Ok(enabled) => {
                                app.set_plan_mode_enabled(enabled);
                                push_system_message(
                                    &mut app,
                                    if enabled {
                                        "Plan mode enabled."
                                    } else {
                                        "Plan mode disabled."
                                    },
                                );
                            }
                            Err(err) => push_system_message(
                                &mut app,
                                format!("Failed to toggle plan mode: {}", err),
                            ),
                        }
                    }
                    // Tab: complete selected suggestion
                    KeyCode::Tab if app.has_suggestions() => {
                        app.complete_suggestion();
                        app.update_suggestions();
                    }
                    KeyCode::Tab => {
                        let input = app.take_input();
                        let images = app.take_images();
                        app.update_suggestions();
                        let queued = QueuedTurn::new(input.clone(), images);
                        let trimmed = input.trim_start();
                        if queued.is_empty() || trimmed.starts_with('!') || trimmed.starts_with('/')
                        {
                            app.input = input;
                            app.cursor_pos = app.input.len();
                            app.pending_images = queued.images;
                            continue;
                        }
                        if app.running {
                            app.queue_turn(queued.clone());
                            app.preview_queued_turn(&queued, false);
                            continue;
                        }
                        if runtime.is_none() {
                            push_system_message(
                                &mut app,
                                "Runtime is still finalizing the previous turn. Please retry in a moment.",
                            );
                            app.input = queued.text;
                            app.cursor_pos = app.input.len();
                            app.pending_images = queued.images;
                            continue;
                        }

                        let (items, image_blobs) =
                            build_items_from_editor_input(&queued.text, queued.images);
                        let turn_input = make_turn_input(&mut app, items, image_blobs);
                        send_user_message(
                            queued.text.clone(),
                            turn_input.clone(),
                            &mut app,
                            logger,
                            &mut runtime,
                            &mut history,
                            &mut runtime_return_rx,
                            &mut cancel_token,
                            &mut active_stream_id,
                            &app_tx,
                        );
                        last_turn = Some(TurnReplayPayload {
                            display_input: queued.text,
                            turn_input,
                            execution_mode: current_execution_mode,
                        });
                    }
                    // Up/Down: navigate suggestions when popup is visible
                    KeyCode::Up if app.has_suggestions() => {
                        app.suggestion_up();
                    }
                    KeyCode::Down if app.has_suggestions() => {
                        app.suggestion_down();
                    }
                    KeyCode::Enter => {
                        // Shift+Enter or Alt+Enter → insert newline
                        if key.modifiers.contains(KeyModifiers::SHIFT)
                            || key.modifiers.contains(KeyModifiers::ALT)
                        {
                            app.insert_char('\n');
                            app.update_suggestions();
                            continue;
                        }

                        let input = app.take_input();
                        let images = app.take_images();
                        app.update_suggestions();
                        let queued = QueuedTurn::new(input.clone(), images);
                        if queued.is_empty() {
                            continue;
                        }

                        if app.running {
                            let trimmed = queued.text.trim_start();
                            if trimmed.starts_with('/') || trimmed.starts_with('!') {
                                push_system_message(
                                    &mut app,
                                    "Host commands cannot be injected into the active turn. Wait for completion or use `Tab` to queue a normal next turn.",
                                );
                                app.input = queued.text;
                                app.cursor_pos = app.input.len();
                                app.pending_images = queued.images;
                                continue;
                            }
                            if !queued.images.is_empty() {
                                push_system_message(
                                    &mut app,
                                    "Current-turn injection does not support images yet. Use `Tab` to queue an image-bearing next turn.",
                                );
                                app.input = queued.text;
                                app.cursor_pos = app.input.len();
                                app.pending_images = queued.images;
                                continue;
                            }
                            let injection = PluginMessage {
                                role: MessageRole::User,
                                content: queued.text.clone(),
                            };
                            match turn_injection_bridge.enqueue(vec![injection]) {
                                Ok(()) => {
                                    app.queue_pending_steer(queued.clone());
                                    app.preview_queued_turn(&queued, true);
                                }
                                Err(err) => {
                                    push_system_message(
                                        &mut app,
                                        format!("Failed to queue current-turn injection: {}", err),
                                    );
                                    app.input = queued.text;
                                    app.cursor_pos = app.input.len();
                                    app.pending_images = queued.images;
                                }
                            }
                            continue;
                        }
                        if runtime.is_none() {
                            push_system_message(
                                &mut app,
                                "Runtime is still finalizing the previous turn. Please retry in a moment.",
                            );
                            app.input = queued.text;
                            app.cursor_pos = app.input.len();
                            app.pending_images = queued.images;
                            continue;
                        }

                        // Shell escape: !command
                        if let Some(cmd_str) = queued.text.strip_prefix('!') {
                            let cmd_str = cmd_str.trim();
                            if !cmd_str.is_empty() {
                                app.blocks
                                    .push(DisplayBlock::UserInput(queued.text.clone()));
                                app.invalidate_height_cache();

                                use tokio::process::Command as TokioCommand;
                                let result = tokio::time::timeout(
                                    std::time::Duration::from_secs(30),
                                    TokioCommand::new("bash").arg("-c").arg(cmd_str).output(),
                                )
                                .await;

                                match result {
                                    Ok(Ok(output)) => {
                                        let stdout =
                                            String::from_utf8_lossy(&output.stdout).to_string();
                                        let stderr =
                                            String::from_utf8_lossy(&output.stderr).to_string();
                                        let error = if !output.status.success() {
                                            let mut err = stderr.clone();
                                            if let Some(code) = output.status.code() {
                                                if !err.is_empty() && !err.ends_with('\n') {
                                                    err.push('\n');
                                                }
                                                err.push_str(&format!("[exit code: {}]", code));
                                            }
                                            Some(err)
                                        } else if !stderr.is_empty() {
                                            Some(stderr)
                                        } else {
                                            None
                                        };
                                        app.blocks.push(DisplayBlock::ShellOutput {
                                            command: cmd_str.to_string(),
                                            output: stdout.trim_end().to_string(),
                                            error: error.map(|e| e.trim_end().to_string()),
                                        });
                                    }
                                    Ok(Err(e)) => {
                                        app.blocks.push(DisplayBlock::Error(format!(
                                            "Failed to run '{}': {}",
                                            cmd_str, e
                                        )));
                                    }
                                    Err(_) => {
                                        app.blocks.push(DisplayBlock::Error(format!(
                                            "Command '{}' timed out after 30s. Try a narrower command or run it in smaller steps.",
                                            cmd_str
                                        )));
                                    }
                                }
                                app.invalidate_height_cache();
                                app.scroll_to_bottom();
                            }
                            continue;
                        }

                        // Try slash command
                        if let Some(cmd) = command::parse(&queued.text, &app.skills) {
                            match cmd {
                                command::Command::Exit => break,
                                command::Command::Clear => {
                                    app.clear();
                                    app.set_model_variant(current_model_variant.clone());
                                    history.clear();
                                    turn_counter = 0;
                                    last_turn = None;
                                    app.token_usage = TokenUsage::default();
                                    active_stream_id = active_stream_id.wrapping_add(1);
                                    if let Some(rt) = runtime.as_mut() {
                                        let _ = rt.reset_session().await;
                                        rt.set_state(AgentStateEnvelope {
                                            agent_id: "root".to_string(),
                                            context_folding: current_context_folding,
                                            messages: history.clone(),
                                            iteration: turn_counter,
                                            token_usage: app.token_usage.clone(),
                                            last_prompt_usage: None,
                                            execution_mode: current_execution_mode,
                                            task_state: None,
                                            subagent_state: None,
                                            replay_manifest: None,
                                            plugin_snapshot: None,
                                            repl_snapshot: None,
                                        });
                                        match rt.session_manager() {
                                            Ok(manager) => session_manager = manager,
                                            Err(err) => push_system_message(
                                                &mut app,
                                                format!(
                                                    "Failed to refresh session manager: {}",
                                                    err
                                                ),
                                            ),
                                        }
                                        sync_plan_mode(
                                            &mut app,
                                            &plugin_host,
                                            Arc::clone(&session_manager),
                                        )
                                        .await;
                                        pending_clear_after_return = false;
                                    } else {
                                        // Runtime is still being reclaimed from a just-finished
                                        // turn; clear state as soon as it returns.
                                        pending_clear_after_return = true;
                                    }
                                }
                                command::Command::Version => {
                                    push_system_message(&mut app, version_text());
                                }
                                command::Command::Info => {
                                    let model = app.model.clone();
                                    let context_window = app.context_window;
                                    let cwd = app.cwd.clone();
                                    let session_name = app.session_name.clone();
                                    push_system_message(
                                        &mut app,
                                        info_text(
                                            &provider,
                                            &model,
                                            current_model_variant.as_deref(),
                                            current_execution_mode,
                                            context_window,
                                            dynamic_tools.definitions().len(),
                                            &toolset_hash,
                                            current_context_folding,
                                            &cwd,
                                            Some(&session_name),
                                        ),
                                    );
                                }
                                command::Command::Model(new_model) => {
                                    let Some(new_model) = new_model else {
                                        let mut lines = vec![
                                            format!("Current model: `{}`", app.model),
                                            format!("Provider: {}", provider.label()),
                                        ];
                                        lines.extend(variant_lines(
                                            &provider,
                                            &app.model,
                                            current_model_variant.as_deref(),
                                        ));
                                        if let Some(window) = app.context_window {
                                            lines.push(format!("Context window: {}", window));
                                        }
                                        lines.push("Usage: `/model <name>`".to_string());
                                        lines.push("Use `/variant` to inspect or change the active variant.".to_string());
                                        push_system_message(&mut app, lines.join("\n"));
                                        continue;
                                    };
                                    let selection = match parse_model_selection(&new_model) {
                                        Ok(s) => s,
                                        Err(e) => {
                                            push_system_message(
                                                &mut app,
                                                format!("Invalid model input: {}", e),
                                            );
                                            continue;
                                        }
                                    };
                                    if let Err(e) = validate_model_selection(&provider, &selection)
                                    {
                                        push_system_message(
                                            &mut app,
                                            format!("Model rejected: {}", e),
                                        );
                                        continue;
                                    }
                                    let resolved_model_spec = match resolve_model_selection(
                                        &provider,
                                        &selection,
                                        model_catalog.as_ref(),
                                    ) {
                                        Ok(spec) => spec,
                                        Err(err) => {
                                            push_system_message(
                                                &mut app,
                                                format!("Model rejected: {}", err),
                                            );
                                            continue;
                                        }
                                    };
                                    let model_variant = provider
                                        .default_model_variant(&selection.model)
                                        .map(str::to_string);
                                    if let Some(rt) = runtime.as_mut() {
                                        rt.update_session_config(
                                            None,
                                            Some(selection.model.clone()),
                                            Some(model_variant.clone()),
                                            Some(resolved_model_spec.context_window() as usize),
                                            None,
                                        )
                                        .await;
                                    }
                                    current_model_variant = model_variant;
                                    app.context_window = Some(resolved_model_spec.context_window());
                                    app.context_usage_excludes_cached_input =
                                        provider.input_usage_excludes_cached_tokens();
                                    app.model = selection.model.clone();
                                    app.set_model_variant(current_model_variant.clone());
                                    let mut msg = format!("Model set to `{}`", app.model);
                                    if let Some(variant) = current_model_variant.as_deref() {
                                        msg.push_str(&format!("\nVariant reset to `{}`", variant));
                                        msg.push_str("\nUse `/variant` to pick a different provider-native preset.");
                                    } else {
                                        msg.push_str(
                                            "\nThis model does not expose configurable variants.",
                                        );
                                    }
                                    if let Some(window) = app.context_window {
                                        msg.push_str(&format!("\nContext window: {}", window));
                                    }
                                    push_system_message(&mut app, msg);
                                }
                                command::Command::Variant(new_variant) => {
                                    let Some(new_variant) = new_variant else {
                                        let mut lines = vec![
                                            format!("Current model: `{}`", app.model),
                                            format!("Provider: {}", provider.label()),
                                        ];
                                        lines.extend(variant_lines(
                                            &provider,
                                            &app.model,
                                            current_model_variant.as_deref(),
                                        ));
                                        push_system_message(&mut app, lines.join("\n"));
                                        continue;
                                    };
                                    let variant = match resolve_model_variant(
                                        &provider,
                                        &app.model,
                                        Some(new_variant.as_str()),
                                    ) {
                                        Ok(variant) => variant,
                                        Err(err) => {
                                            push_system_message(
                                                &mut app,
                                                format!("Variant rejected: {}", err),
                                            );
                                            continue;
                                        }
                                    };
                                    if let Some(rt) = runtime.as_mut() {
                                        rt.update_session_config(
                                            None,
                                            None,
                                            Some(variant.clone()),
                                            None,
                                            None,
                                        )
                                        .await;
                                    }
                                    current_model_variant = variant;
                                    app.set_model_variant(current_model_variant.clone());
                                    let mut lines = vec![format!("Model: `{}`", app.model)];
                                    if let Some(variant) = current_model_variant.as_deref() {
                                        lines.push(format!("Variant set to `{}`", variant));
                                    } else {
                                        lines.push(
                                            "Variant reset to provider default `(none)`."
                                                .to_string(),
                                        );
                                    }
                                    lines.extend(variant_lines(
                                        &provider,
                                        &app.model,
                                        current_model_variant.as_deref(),
                                    ));
                                    push_system_message(&mut app, lines.join("\n"));
                                }
                                command::Command::Mode(new_mode) => {
                                    let Some(new_mode) = new_mode else {
                                        push_system_message(
                                            &mut app,
                                            format!(
                                                "Current execution mode: `{}`\nThis is locked for the current session.\nStart a new session to use a different mode.\nUsage: `/mode {}`",
                                                execution_mode_label(current_execution_mode),
                                                execution_mode_usage()
                                            ),
                                        );
                                        continue;
                                    };
                                    let new_mode = match parse_execution_mode(&new_mode)
                                        .and_then(ensure_supported_execution_mode)
                                    {
                                        Ok(mode) => mode,
                                        Err(err) => {
                                            push_system_message(
                                                &mut app,
                                                format!("Invalid execution mode: {}", err),
                                            );
                                            continue;
                                        }
                                    };
                                    if new_mode == current_execution_mode {
                                        push_system_message(
                                            &mut app,
                                            format!(
                                                "Execution mode is already `{}`.\nThis is locked for the current session.",
                                                execution_mode_label(current_execution_mode)
                                            ),
                                        );
                                    } else {
                                        push_system_message(
                                            &mut app,
                                            format!(
                                                "Execution mode is locked for the current session (`{}`).\nStart a new session with `--mode {}` to use `{}`.",
                                                execution_mode_label(current_execution_mode),
                                                execution_mode_label(new_mode),
                                                execution_mode_label(new_mode)
                                            ),
                                        );
                                    }
                                }
                                command::Command::ChangeProvider => {
                                    paused.store(true, Ordering::Relaxed);
                                    let _ = crossterm::execute!(
                                        std::io::stdout(),
                                        crossterm::event::DisableFocusChange
                                    );
                                    if !args.no_mouse {
                                        let _ = crossterm::execute!(
                                            std::io::stdout(),
                                            crossterm::event::DisableMouseCapture
                                        );
                                    }
                                    cleanup_terminal();
                                    let existing_cfg = LashConfig::load();
                                    let setup_result =
                                        setup::run_setup_with_existing(existing_cfg.as_ref()).await;
                                    terminal = ratatui::init();
                                    configure_terminal_ui(args.no_mouse)?;
                                    paused.store(false, Ordering::Relaxed);

                                    match setup_result {
                                        Ok(mut new_cfg) => {
                                            let previous_kind = provider.kind();
                                            if let Err(e) =
                                                new_cfg.active_provider_mut().ensure_fresh().await
                                            {
                                                push_system_message(
                                                    &mut app,
                                                    format!(
                                                        "Provider setup completed, but token refresh failed: {}",
                                                        e
                                                    ),
                                                );
                                                continue;
                                            }
                                            if let Err(e) = new_cfg.save() {
                                                push_system_message(
                                                    &mut app,
                                                    format!(
                                                        "Provider updated, but saving config failed: {}",
                                                        e
                                                    ),
                                                );
                                            }
                                            provider = new_cfg.active_provider().clone();
                                            if let Err(err) = model_catalog
                                                .refresh_if_stale(
                                                    lash_core::model_info::DEFAULT_REFRESH_INTERVAL,
                                                )
                                                .await
                                            {
                                                push_system_message(
                                                    &mut app,
                                                    format!(
                                                        "Warning: failed to refresh models.dev catalog: {}",
                                                        err
                                                    ),
                                                );
                                            }
                                            let new_model = provider.default_model().to_string();
                                            let selection = match parse_model_selection(&new_model)
                                            {
                                                Ok(s) => s,
                                                Err(e) => {
                                                    push_system_message(
                                                        &mut app,
                                                        format!(
                                                            "Provider default model is invalid: {}",
                                                            e
                                                        ),
                                                    );
                                                    continue;
                                                }
                                            };
                                            if let Err(e) =
                                                validate_model_selection(&provider, &selection)
                                            {
                                                push_system_message(
                                                    &mut app,
                                                    format!(
                                                        "Provider default model failed validation: {}",
                                                        e
                                                    ),
                                                );
                                                continue;
                                            }
                                            let resolved_model_spec = match resolve_model_selection(
                                                &provider,
                                                &selection,
                                                model_catalog.as_ref(),
                                            ) {
                                                Ok(spec) => spec,
                                                Err(err) => {
                                                    push_system_message(
                                                        &mut app,
                                                        format!(
                                                            "Provider default model failed validation: {}",
                                                            err
                                                        ),
                                                    );
                                                    continue;
                                                }
                                            };
                                            let model_variant = provider
                                                .default_model_variant(&selection.model)
                                                .map(str::to_string);
                                            if let Some(rt) = runtime.as_mut() {
                                                rt.update_session_config(
                                                    Some(provider.clone()),
                                                    Some(selection.model.clone()),
                                                    Some(model_variant.clone()),
                                                    Some(resolved_model_spec.context_window()
                                                        as usize),
                                                    None,
                                                )
                                                .await;
                                            }
                                            current_model_variant = model_variant;
                                            app.context_window =
                                                Some(resolved_model_spec.context_window());
                                            app.context_usage_excludes_cached_input =
                                                provider.input_usage_excludes_cached_tokens();
                                            app.model = selection.model.clone();
                                            app.set_model_variant(current_model_variant.clone());
                                            let saved_kinds = new_cfg
                                                .provider_kinds()
                                                .into_iter()
                                                .map(ProviderKind::cli_label)
                                                .collect::<Vec<_>>()
                                                .join(", ");
                                            push_system_message(
                                                &mut app,
                                                format!(
                                                    "Provider {}: {}\nSaved providers: {}\nModel set to default: `{}`",
                                                    if provider.kind() == previous_kind {
                                                        "reauthenticated"
                                                    } else {
                                                        "switched"
                                                    },
                                                    provider.label(),
                                                    saved_kinds,
                                                    selection.model
                                                ),
                                            );
                                        }
                                        Err(e) => {
                                            let msg = e.to_string();
                                            if msg.contains("Setup cancelled") {
                                                push_system_message(
                                                    &mut app,
                                                    "Provider setup cancelled. Current provider unchanged.",
                                                );
                                            } else {
                                                push_system_message(
                                                    &mut app,
                                                    format!(
                                                        "Provider setup failed: {}. Current provider unchanged.",
                                                        msg
                                                    ),
                                                );
                                            }
                                        }
                                    }
                                }
                                command::Command::Logout => {
                                    let active_kind = provider.kind();
                                    match LashConfig::load() {
                                        Some(mut cfg) => {
                                            if !cfg.has_provider(active_kind) {
                                                push_system_message(
                                                    &mut app,
                                                    "The active provider is not stored on disk.",
                                                );
                                                continue;
                                            }
                                            if cfg.provider_count() == 1 {
                                                match LashConfig::clear() {
                                                    Ok(()) => push_system_message(
                                                        &mut app,
                                                        format!(
                                                            "Removed stored credentials for {}.\n\nThis running session may continue using in-memory credentials.\nUse `/provider` or `/login` to sign in again without restarting.",
                                                            active_kind.cli_label()
                                                        ),
                                                    ),
                                                    Err(e) => push_system_message(
                                                        &mut app,
                                                        format!(
                                                            "Failed to remove credentials: {}",
                                                            e
                                                        ),
                                                    ),
                                                }
                                            } else {
                                                cfg.remove_provider(active_kind);
                                                let next_kind = cfg.active_provider_kind();
                                                match cfg.save() {
                                                    Ok(()) => push_system_message(
                                                        &mut app,
                                                        format!(
                                                            "Removed stored credentials for {}.\nNew sessions will default to {}.\n\nThis running session may continue using in-memory credentials.",
                                                            active_kind.cli_label(),
                                                            next_kind.cli_label()
                                                        ),
                                                    ),
                                                    Err(e) => push_system_message(
                                                        &mut app,
                                                        format!(
                                                            "Failed to save updated provider registry: {}",
                                                            e
                                                        ),
                                                    ),
                                                }
                                            }
                                        }
                                        None => push_system_message(
                                            &mut app,
                                            "No stored provider credentials found.",
                                        ),
                                    }
                                }
                                command::Command::Retry => {
                                    if let Some(previous) = last_turn.clone() {
                                        if let Err(e) = apply_pending_reconfigure(
                                            &dynamic_tools,
                                            &mut desired_dynamic,
                                            &mut pending_reconfigure,
                                            &mut runtime,
                                        )
                                        .await
                                        {
                                            push_system_message(
                                                &mut app,
                                                format!(
                                                    "Pending runtime reconfigure failed; retry blocked: {}",
                                                    e
                                                ),
                                            );
                                            continue;
                                        }
                                        toolset_hash = hash12(
                                            &serde_json::to_vec(&dynamic_tools.definitions())
                                                .unwrap_or_else(|_| b"[]".to_vec()),
                                        );
                                        current_execution_mode = previous.execution_mode;
                                        send_user_message(
                                            previous.display_input.clone(),
                                            previous.turn_input.clone(),
                                            &mut app,
                                            logger,
                                            &mut runtime,
                                            &mut history,
                                            &mut runtime_return_rx,
                                            &mut cancel_token,
                                            &mut active_stream_id,
                                            &app_tx,
                                        );
                                    } else {
                                        push_system_message(
                                            &mut app,
                                            "No previous turn payload to retry yet.",
                                        );
                                    }
                                }
                                command::Command::Controls => {
                                    push_system_message(&mut app, controls_text());
                                }
                                command::Command::Fork(prompt) => {
                                    let Some(rt) = runtime.as_mut() else {
                                        push_system_message(
                                            &mut app,
                                            "Runtime is not available to fork right now.",
                                        );
                                        continue;
                                    };
                                    let current_dynamic_state = dynamic_tools.export_state();
                                    match fork::fork_current_session(
                                        rt,
                                        logger,
                                        &provider,
                                        &app.model,
                                        app.context_window.expect(
                                            "app context_window must be set before forking",
                                        ),
                                        current_model_variant.as_deref(),
                                        &toolset_hash,
                                        &current_dynamic_state,
                                    )
                                    .await
                                    {
                                        Ok((child_filename, child_session_name)) => {
                                            let exe = match std::env::current_exe() {
                                                Ok(exe) => exe,
                                                Err(err) => {
                                                    push_system_message(
                                                        &mut app,
                                                        format!(
                                                            "Fork created but launcher lookup failed: {}",
                                                            err
                                                        ),
                                                    );
                                                    continue;
                                                }
                                            };
                                            let mut child_args = vec![
                                                "--resume".to_string(),
                                                child_filename.clone(),
                                            ];
                                            if let Some(prompt) = prompt
                                                .as_deref()
                                                .map(str::trim)
                                                .filter(|prompt| !prompt.is_empty())
                                            {
                                                child_args.push("--resume-prompt".to_string());
                                                child_args.push(prompt.to_string());
                                            }
                                            match fork::spawn_in_new_terminal(&exe, &child_args) {
                                                Ok(()) => push_system_message(
                                                    &mut app,
                                                    format!(
                                                        "Forked into `{}` ({})",
                                                        child_session_name, child_filename
                                                    ),
                                                ),
                                                Err(err) => push_system_message(
                                                    &mut app,
                                                    format!(
                                                        "Fork created but launch failed: {}",
                                                        err
                                                    ),
                                                ),
                                            }
                                        }
                                        Err(err) => push_system_message(
                                            &mut app,
                                            format!("Fork failed: {}", err),
                                        ),
                                    }
                                }
                                command::Command::Help => {
                                    let help = help_text(&app.skills);
                                    push_system_message(&mut app, help);
                                }
                                command::Command::Resume(name) => {
                                    if let Some(filename) = name {
                                        match resume::load_resumed_session(
                                            &filename,
                                            &mut app,
                                            &mut history,
                                            &mut runtime,
                                            &mut turn_counter,
                                            &mut current_execution_mode,
                                            &provider,
                                            &mut current_model_variant,
                                            &dynamic_tools,
                                            &mut desired_dynamic,
                                            model_catalog.as_ref(),
                                        )
                                        .await
                                        {
                                            Ok(()) => {
                                                last_turn = None;
                                                if let Some(rt) = runtime.as_ref() {
                                                    match rt.session_manager() {
                                                        Ok(manager) => session_manager = manager,
                                                        Err(err) => push_system_message(
                                                            &mut app,
                                                            format!(
                                                                "Failed to refresh session manager: {}",
                                                                err
                                                            ),
                                                        ),
                                                    }
                                                }
                                                sync_plan_mode(
                                                    &mut app,
                                                    &plugin_host,
                                                    Arc::clone(&session_manager),
                                                )
                                                .await;
                                                toolset_hash = hash12(
                                                    &serde_json::to_vec(
                                                        &dynamic_tools.definitions(),
                                                    )
                                                    .unwrap_or_else(|_| b"[]".to_vec()),
                                                );
                                            }
                                            Err(err) => {
                                                app.blocks.push(DisplayBlock::SystemMessage(err));
                                                app.invalidate_height_cache();
                                                app.scroll_to_bottom();
                                            }
                                        }
                                    } else {
                                        // No arg — open session picker
                                        let mut sessions = session_log::list_sessions();
                                        sessions.retain(|s| s.session_id != logger.session_id);
                                        if sessions.is_empty() {
                                            app.blocks.push(DisplayBlock::SystemMessage(
                                                "No sessions found.".to_string(),
                                            ));
                                            app.invalidate_height_cache();
                                            app.scroll_to_bottom();
                                        } else {
                                            sessions.truncate(50);
                                            app.session_picker = sessions;
                                            app.session_picker_idx = 0;
                                        }
                                    }
                                }
                                command::Command::Tools(raw) => {
                                    let raw = raw.unwrap_or_default();
                                    let raw_trim = raw.trim();
                                    if raw_trim.is_empty() {
                                        let active = dynamic_tools.export_state();
                                        let mut lines = vec![
                                            format!(
                                                "Dynamic tools (generation {}):",
                                                active.base_generation
                                            ),
                                            format!(
                                                "Pending reconfigure: {}",
                                                if pending_reconfigure { "yes" } else { "no" }
                                            ),
                                        ];
                                        for (name, spec) in &desired_dynamic.tools {
                                            let enabled = desired_dynamic
                                                .profile
                                                .enabled_tools
                                                .contains(name);
                                            lines.push(format!(
                                                "  - {} [{}] adapter={}{}",
                                                name,
                                                spec.definition.returns,
                                                spec.adapter_id,
                                                if enabled { " (explicitly enabled)" } else { "" }
                                            ));
                                        }
                                        if desired_dynamic.tools.is_empty() {
                                            lines.push("  (none)".to_string());
                                        }
                                        push_system_message(&mut app, lines.join("\n"));
                                        continue;
                                    }

                                    let mut parts = raw_trim.split_whitespace();
                                    let sub = parts.next().unwrap_or_default();
                                    match sub {
                                        "add" => {
                                            let mut add_parts = raw_trim.splitn(4, ' ');
                                            let _ = add_parts.next();
                                            let Some(name) = add_parts.next() else {
                                                push_system_message(
                                                    &mut app,
                                                    "Usage: /tools add <name> <handler> [description]",
                                                );
                                                continue;
                                            };
                                            let Some(handler_id) = add_parts.next() else {
                                                push_system_message(
                                                    &mut app,
                                                    "Usage: /tools add <name> <handler> [description]",
                                                );
                                                continue;
                                            };
                                            let description =
                                                add_parts.next().map(|v| v.trim().to_string());
                                            match register_builtin_tool(
                                                &dynamic_tools,
                                                name,
                                                handler_id,
                                                description,
                                                current_execution_mode,
                                            ) {
                                                Ok(def) => {
                                                    desired_dynamic.tools.insert(
                                                        name.to_string(),
                                                        DynamicToolSpec {
                                                            definition: def,
                                                            adapter_id: "inprocess".to_string(),
                                                        },
                                                    );
                                                    desired_dynamic
                                                        .profile
                                                        .enabled_tools
                                                        .insert(name.to_string());
                                                    pending_reconfigure = true;
                                                    push_system_message(
                                                        &mut app,
                                                        format!(
                                                            "Tool `{}` staged with handler `{}`. Apply with `/reconfigure apply` or send the next turn.",
                                                            name, handler_id
                                                        ),
                                                    );
                                                }
                                                Err(e) => push_system_message(&mut app, e),
                                            }
                                        }
                                        "rm" | "remove" => {
                                            let Some(name) = parts.next() else {
                                                push_system_message(
                                                    &mut app,
                                                    "Usage: /tools rm <name>",
                                                );
                                                continue;
                                            };
                                            if desired_dynamic.tools.remove(name).is_some() {
                                                desired_dynamic.profile.enabled_tools.remove(name);
                                                for cap in
                                                    desired_dynamic.capability_defs.values_mut()
                                                {
                                                    cap.tool_names.remove(name);
                                                }
                                                pending_reconfigure = true;
                                                push_system_message(
                                                    &mut app,
                                                    format!("Tool `{name}` staged for removal."),
                                                );
                                            } else {
                                                push_system_message(
                                                    &mut app,
                                                    format!("Tool `{name}` not found."),
                                                );
                                            }
                                        }
                                        "update" => {
                                            let mut update_parts = raw_trim.splitn(3, ' ');
                                            let _ = update_parts.next();
                                            let Some(name) = update_parts.next() else {
                                                push_system_message(
                                                    &mut app,
                                                    "Usage: /tools update <name> key=value ...",
                                                );
                                                continue;
                                            };
                                            let kv_raw = update_parts.next().unwrap_or_default();
                                            let kv = parse_kv_args(kv_raw);
                                            let Some(spec) = desired_dynamic.tools.get_mut(name)
                                            else {
                                                push_system_message(
                                                    &mut app,
                                                    format!("Tool `{name}` not found."),
                                                );
                                                continue;
                                            };
                                            if let Some(desc) = kv.get("description") {
                                                spec.definition.set_description_for(
                                                    current_execution_mode,
                                                    desc.clone(),
                                                );
                                            }
                                            if let Some(returns) = kv.get("returns") {
                                                spec.definition.returns = returns.clone();
                                            }
                                            if let Some(inject) = kv.get("inject_into_prompt") {
                                                spec.definition.inject_into_prompt =
                                                    inject == "true";
                                            }
                                            pending_reconfigure = true;
                                            push_system_message(
                                                &mut app,
                                                format!("Tool `{name}` staged for update."),
                                            );
                                        }
                                        _ => push_system_message(
                                            &mut app,
                                            "Unknown /tools subcommand. Try: add, rm, update",
                                        ),
                                    }
                                }
                                command::Command::Caps(raw) => {
                                    let raw = raw.unwrap_or_default();
                                    let raw_trim = raw.trim();
                                    if raw_trim.is_empty() {
                                        let mut lines = vec![
                                            "Dynamic capabilities:".to_string(),
                                            format!(
                                                "Enabled ids: {}",
                                                desired_dynamic
                                                    .profile
                                                    .enabled_capabilities
                                                    .iter()
                                                    .cloned()
                                                    .collect::<Vec<_>>()
                                                    .join(", ")
                                            ),
                                        ];
                                        for (id, cap) in &desired_dynamic.capability_defs {
                                            lines.push(format!(
                                                "  - {} ({}) tools=[{}]",
                                                id,
                                                cap.name,
                                                cap.tool_names
                                                    .iter()
                                                    .cloned()
                                                    .collect::<Vec<_>>()
                                                    .join(", ")
                                            ));
                                        }
                                        push_system_message(&mut app, lines.join("\n"));
                                        continue;
                                    }

                                    let mut parts = raw_trim.split_whitespace();
                                    let sub = parts.next().unwrap_or_default();
                                    match sub {
                                        "add" => {
                                            let mut add_parts = raw_trim.splitn(3, ' ');
                                            let _ = add_parts.next();
                                            let Some(id) = add_parts.next() else {
                                                push_system_message(
                                                    &mut app,
                                                    "Usage: /caps add <id> name=<n> tools=a,b helpers=h1,h2 prompt=<text>",
                                                );
                                                continue;
                                            };
                                            let kv =
                                                parse_kv_args(add_parts.next().unwrap_or_default());
                                            let name = kv
                                                .get("name")
                                                .cloned()
                                                .unwrap_or_else(|| id.to_string());
                                            let tools = kv
                                                .get("tools")
                                                .map_or_else(BTreeSet::new, |v| parse_csv(v));
                                            let helpers = kv
                                                .get("helpers")
                                                .map_or_else(BTreeSet::new, |v| parse_csv(v));
                                            let prompt = kv.get("prompt").cloned();
                                            desired_dynamic.capability_defs.insert(
                                                id.to_string(),
                                                DynamicCapabilityDef {
                                                    id: id.to_string(),
                                                    name,
                                                    description: kv
                                                        .get("description")
                                                        .cloned()
                                                        .unwrap_or_default(),
                                                    prompt_section: prompt,
                                                    helper_bindings: helpers,
                                                    tool_names: tools,
                                                    enabled_by_default: kv
                                                        .get("enabled_by_default")
                                                        .map(|v| v == "true")
                                                        .unwrap_or(false),
                                                },
                                            );
                                            pending_reconfigure = true;
                                            push_system_message(
                                                &mut app,
                                                format!("Capability `{id}` staged."),
                                            );
                                        }
                                        "rm" | "remove" => {
                                            let Some(id) = parts.next() else {
                                                push_system_message(
                                                    &mut app,
                                                    "Usage: /caps rm <id>",
                                                );
                                                continue;
                                            };
                                            desired_dynamic.capability_defs.remove(id);
                                            desired_dynamic.profile.enabled_capabilities.remove(id);
                                            pending_reconfigure = true;
                                            push_system_message(
                                                &mut app,
                                                format!("Capability `{id}` staged for removal."),
                                            );
                                        }
                                        "enable" => {
                                            let Some(id) = parts.next() else {
                                                push_system_message(
                                                    &mut app,
                                                    "Usage: /caps enable <id>",
                                                );
                                                continue;
                                            };
                                            desired_dynamic
                                                .profile
                                                .enabled_capabilities
                                                .insert(id.to_string());
                                            pending_reconfigure = true;
                                            push_system_message(
                                                &mut app,
                                                format!("Capability `{id}` staged for enable."),
                                            );
                                        }
                                        "disable" => {
                                            let Some(id) = parts.next() else {
                                                push_system_message(
                                                    &mut app,
                                                    "Usage: /caps disable <id>",
                                                );
                                                continue;
                                            };
                                            desired_dynamic.profile.enabled_capabilities.remove(id);
                                            pending_reconfigure = true;
                                            push_system_message(
                                                &mut app,
                                                format!("Capability `{id}` staged for disable."),
                                            );
                                        }
                                        "tool-enable" => {
                                            let Some(name) = parts.next() else {
                                                push_system_message(
                                                    &mut app,
                                                    "Usage: /caps tool-enable <tool>",
                                                );
                                                continue;
                                            };
                                            desired_dynamic
                                                .profile
                                                .enabled_tools
                                                .insert(name.to_string());
                                            pending_reconfigure = true;
                                            push_system_message(
                                                &mut app,
                                                format!(
                                                    "Tool `{name}` staged as explicitly enabled."
                                                ),
                                            );
                                        }
                                        "tool-disable" => {
                                            let Some(name) = parts.next() else {
                                                push_system_message(
                                                    &mut app,
                                                    "Usage: /caps tool-disable <tool>",
                                                );
                                                continue;
                                            };
                                            desired_dynamic.profile.enabled_tools.remove(name);
                                            pending_reconfigure = true;
                                            push_system_message(
                                                &mut app,
                                                format!(
                                                    "Tool `{name}` staged as explicitly disabled."
                                                ),
                                            );
                                        }
                                        _ => push_system_message(
                                            &mut app,
                                            "Unknown /caps subcommand. Try: add, rm, enable, disable, tool-enable, tool-disable",
                                        ),
                                    }
                                }
                                command::Command::Reconfigure(raw) => {
                                    let action = raw.unwrap_or_else(|| "status".to_string());
                                    match action.trim() {
                                        "" | "status" => {
                                            push_system_message(
                                                &mut app,
                                                format!(
                                                    "Reconfigure status: pending={} current_generation={} base_generation={}",
                                                    pending_reconfigure,
                                                    dynamic_tools.generation(),
                                                    desired_dynamic.base_generation
                                                ),
                                            );
                                        }
                                        "clear" => {
                                            desired_dynamic = dynamic_tools.export_state();
                                            pending_reconfigure = false;
                                            push_system_message(
                                                &mut app,
                                                "Cleared pending dynamic runtime changes.",
                                            );
                                        }
                                        "apply" => {
                                            match apply_pending_reconfigure(
                                                &dynamic_tools,
                                                &mut desired_dynamic,
                                                &mut pending_reconfigure,
                                                &mut runtime,
                                            )
                                            .await
                                            {
                                                Ok(generation) => {
                                                    toolset_hash = hash12(
                                                        &serde_json::to_vec(
                                                            &dynamic_tools.definitions(),
                                                        )
                                                        .unwrap_or_else(|_| b"[]".to_vec()),
                                                    );
                                                    push_system_message(
                                                        &mut app,
                                                        format!(
                                                            "Dynamic runtime reconfigured successfully (generation {}).",
                                                            generation
                                                        ),
                                                    )
                                                }
                                                Err(e) => push_system_message(
                                                    &mut app,
                                                    format!("Reconfigure failed: {e}"),
                                                ),
                                            }
                                        }
                                        _ => push_system_message(
                                            &mut app,
                                            "Unknown /reconfigure action. Try: status, apply, clear",
                                        ),
                                    }
                                }
                                command::Command::Skills => {
                                    // Refresh from disk so new/removed skills are picked up
                                    app.skills = skill::SkillRegistry::load();
                                    let items: Vec<(String, String)> = app
                                        .skills
                                        .iter()
                                        .map(|s| (s.name.clone(), s.description.clone()))
                                        .collect();
                                    if items.is_empty() {
                                        app.blocks.push(DisplayBlock::SystemMessage(
                                            "No skills found.\n\
                                             Add skill directories to ~/.lash/skills/ or .agents/lash/skills/\n\
                                             Each skill is a directory with a SKILL.md file."
                                                .to_string(),
                                        ));
                                        app.invalidate_height_cache();
                                        app.scroll_to_bottom();
                                    } else {
                                        app.skill_picker = items;
                                        app.skill_picker_idx = 0;
                                    }
                                }
                                command::Command::Skill(name, args) => {
                                    if app.skills.get(&name).is_some() {
                                        if let Err(e) = apply_pending_reconfigure(
                                            &dynamic_tools,
                                            &mut desired_dynamic,
                                            &mut pending_reconfigure,
                                            &mut runtime,
                                        )
                                        .await
                                        {
                                            push_system_message(
                                                &mut app,
                                                format!(
                                                    "Pending runtime reconfigure failed; skill turn blocked: {}",
                                                    e
                                                ),
                                            );
                                            continue;
                                        }
                                        toolset_hash = hash12(
                                            &serde_json::to_vec(&dynamic_tools.definitions())
                                                .unwrap_or_else(|_| b"[]".to_vec()),
                                        );
                                        // Display original slash command input in UI, but send
                                        // structured SKILL payload to the runtime turn.
                                        let skill_item = InputItem::SkillRef { name, args };
                                        let display_input = input.clone();
                                        let turn_input = make_turn_input(
                                            &mut app,
                                            vec![skill_item],
                                            HashMap::new(),
                                        );
                                        send_user_message(
                                            display_input.clone(),
                                            turn_input.clone(),
                                            &mut app,
                                            logger,
                                            &mut runtime,
                                            &mut history,
                                            &mut runtime_return_rx,
                                            &mut cancel_token,
                                            &mut active_stream_id,
                                            &app_tx,
                                        );
                                        last_turn = Some(TurnReplayPayload {
                                            display_input,
                                            turn_input,
                                            execution_mode: current_execution_mode,
                                        });
                                    }
                                }
                            }
                            continue;
                        }

                        // Handle "quit"/"exit" without slash prefix
                        if input == "quit" || input == "exit" {
                            break;
                        }

                        // Regular user message — send to agent
                        if let Err(e) = apply_pending_reconfigure(
                            &dynamic_tools,
                            &mut desired_dynamic,
                            &mut pending_reconfigure,
                            &mut runtime,
                        )
                        .await
                        {
                            push_system_message(
                                &mut app,
                                format!(
                                    "Pending runtime reconfigure failed; message not sent: {}",
                                    e
                                ),
                            );
                            continue;
                        }
                        toolset_hash = hash12(
                            &serde_json::to_vec(&dynamic_tools.definitions())
                                .unwrap_or_else(|_| b"[]".to_vec()),
                        );
                        let (items, image_blobs) =
                            build_items_from_editor_input(&queued.text, queued.images);
                        let turn_input = make_turn_input(&mut app, items, image_blobs);
                        send_user_message(
                            queued.text.clone(),
                            turn_input.clone(),
                            &mut app,
                            logger,
                            &mut runtime,
                            &mut history,
                            &mut runtime_return_rx,
                            &mut cancel_token,
                            &mut active_stream_id,
                            &app_tx,
                        );
                        last_turn = Some(TurnReplayPayload {
                            display_input: queued.text,
                            turn_input,
                            execution_mode: current_execution_mode,
                        });
                    }
                    KeyCode::Backspace => {
                        app.backspace();
                        app.update_suggestions();
                    }
                    KeyCode::Delete => {
                        app.delete();
                        app.update_suggestions();
                    }
                    KeyCode::Left => app.move_cursor_left(),
                    KeyCode::Right => app.move_cursor_right(),
                    KeyCode::Home => app.move_cursor_home(),
                    KeyCode::End => app.move_cursor_end(),
                    KeyCode::Up => app.history_up(),
                    KeyCode::Down => app.history_down(),
                    KeyCode::Char(c) => {
                        app.insert_char(c);
                        app.update_suggestions();
                    }
                    _ => {}
                }
            }
            AppEvent::Terminal(TermEvent::Mouse(mouse)) => {
                use crossterm::event::{KeyModifiers, MouseEventKind};
                if mouse.modifiers.contains(KeyModifiers::SHIFT) && app.mouse_captured {
                    // Release mouse capture so the terminal handles native
                    // text selection and scrolling while shift is held.
                    let _ = crossterm::execute!(
                        std::io::stdout(),
                        crossterm::event::DisableMouseCapture
                    );
                    app.mouse_captured = false;
                } else {
                    app.dirty = true;
                    match mouse.kind {
                        MouseEventKind::ScrollUp => app.scroll_up(3),
                        MouseEventKind::ScrollDown => {
                            let size = terminal.size()?;
                            let vh = ui::history_viewport_height(&app, size.width, size.height);
                            let vw = size.width as usize;
                            app.scroll_down(3, vh, vw);
                        }
                        _ => {}
                    }
                }
            }
            AppEvent::Terminal(TermEvent::FocusGained) => {
                app.focused = true;
                app.dirty = true;
            }
            AppEvent::Terminal(TermEvent::FocusLost) => {
                app.focused = false;
                app.dirty = true;
            }
            AppEvent::Terminal(_) => {
                // Resize events, etc.
                app.dirty = true;
            }
            AppEvent::Tick => {
                if app.running {
                    app.tick += 1;
                    app.dirty = true;
                }
            }
            AppEvent::Agent { stream_id, event } => {
                if stream_id != active_stream_id {
                    continue;
                }
                app.dirty = true;
                // Intercept Prompt events — set up dialog state instead of passing to handle_agent_event
                if let AgentEvent::Prompt {
                    question,
                    options,
                    response_tx,
                } = event
                {
                    let is_freeform = options.is_empty();
                    app.prompt = Some(app::PromptState {
                        question,
                        options,
                        selected_idx: 0,
                        extra_text: String::new(),
                        extra_cursor: 0,
                        editing_extra: is_freeform, // freeform starts in edit mode
                        response_tx,
                    });
                } else {
                    let is_done = matches!(event, AgentEvent::Done);
                    logger.log_event(&event);
                    app.handle_agent_event(event);
                    if is_done && !app.focused {
                        notify_done();
                    }
                }
            }
            AppEvent::Quit => break,
        }
    }

    // Signal reader thread and tick timer to stop
    stop.store(true, Ordering::Relaxed);

    // Save input history
    app.save_history();
    logger.flush()?;

    Ok(())
}

fn make_turn_input(
    _app: &mut App,
    items: Vec<InputItem>,
    image_blobs: HashMap<String, Vec<u8>>,
) -> TurnInput {
    TurnInput {
        items,
        image_blobs,
        mode: Some(RunMode::Normal),
    }
}

fn parse_kv_args(raw: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for token in raw.split_whitespace() {
        if let Some((k, v)) = token.split_once('=') {
            out.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    out
}

fn parse_csv(raw: &str) -> BTreeSet<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

fn register_builtin_tool(
    dynamic_tools: &Arc<DynamicToolProvider>,
    tool_name: &str,
    handler_id: &str,
    description_override: Option<String>,
    execution_mode: ExecutionMode,
) -> Result<ToolDefinition, String> {
    let adapter = dynamic_tools.inprocess_adapter();
    let def = match handler_id {
        "echo" => {
            let handler: InProcessToolHandler = Arc::new(|args, _progress| {
                Box::pin(async move {
                    let text = args
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    ToolResult::ok(serde_json::json!(text))
                })
            });
            let def = ToolDefinition {
                name: tool_name.to_string(),
                description: vec![ToolText::new(
                    description_override
                        .unwrap_or_else(|| "Echoes back the `text` argument.".to_string()),
                    [execution_mode],
                )],
                params: vec![ToolParam::typed("text", "str")],
                returns: "str".to_string(),
                examples: vec![ToolText::new(
                    format!("{tool_name}(text=\"hello\")"),
                    [execution_mode],
                )],
                hidden: false,
                inject_into_prompt: false,
            };
            adapter.register_tool(def.clone(), handler);
            def
        }
        "time" => {
            let handler: InProcessToolHandler = Arc::new(|_args, _progress| {
                Box::pin(async move {
                    ToolResult::ok(serde_json::json!(chrono::Utc::now().to_rfc3339()))
                })
            });
            let def = ToolDefinition {
                name: tool_name.to_string(),
                description: vec![ToolText::new(
                    description_override.unwrap_or_else(|| {
                        "Returns the current UTC timestamp (RFC3339).".to_string()
                    }),
                    [execution_mode],
                )],
                params: vec![],
                returns: "str".to_string(),
                examples: vec![ToolText::new(format!("{tool_name}()"), [execution_mode])],
                hidden: false,
                inject_into_prompt: false,
            };
            adapter.register_tool(def.clone(), handler);
            def
        }
        "uuid" => {
            let handler: InProcessToolHandler = Arc::new(|_args, _progress| {
                Box::pin(async move {
                    ToolResult::ok(serde_json::json!(uuid::Uuid::new_v4().to_string()))
                })
            });
            let def = ToolDefinition {
                name: tool_name.to_string(),
                description: vec![ToolText::new(
                    description_override
                        .unwrap_or_else(|| "Returns a random UUIDv4 string.".to_string()),
                    [execution_mode],
                )],
                params: vec![],
                returns: "str".to_string(),
                examples: vec![ToolText::new(format!("{tool_name}()"), [execution_mode])],
                hidden: false,
                inject_into_prompt: false,
            };
            adapter.register_tool(def.clone(), handler);
            def
        }
        other => {
            return Err(format!(
                "Unknown handler `{other}`. Supported handlers: echo, time, uuid"
            ));
        }
    };

    Ok(def)
}

async fn apply_pending_reconfigure(
    dynamic_tools: &Arc<DynamicToolProvider>,
    desired_dynamic: &mut DynamicStateSnapshot,
    pending_reconfigure: &mut bool,
    runtime: &mut Option<LashRuntime>,
) -> Result<u64, String> {
    if !*pending_reconfigure {
        return Ok(dynamic_tools.generation());
    }

    let previous = dynamic_tools.export_state();
    let generation = match dynamic_tools.apply_state(desired_dynamic.clone()) {
        Ok(g) => g,
        Err(e) => {
            desired_dynamic.base_generation = dynamic_tools.generation();
            return Err(e.to_string());
        }
    };

    let prompt_caps = agent_capabilities_from_profile(&dynamic_tools.profile());
    if let Some(rt) = runtime.as_mut() {
        rt.set_capabilities(prompt_caps);
        if let Err(e) = rt.refresh_session_execution_surface().await {
            let mut rollback = previous.clone();
            rollback.base_generation = dynamic_tools.generation();
            let _ = dynamic_tools.apply_state(rollback);
            rt.set_capabilities(agent_capabilities_from_profile(&dynamic_tools.profile()));
            let _ = rt.refresh_session_execution_surface().await;
            desired_dynamic.base_generation = dynamic_tools.generation();
            return Err(format!(
                "Failed to apply runtime reconfigure (state rolled back): {e}"
            ));
        }
    }

    *desired_dynamic = dynamic_tools.export_state();
    *pending_reconfigure = false;
    Ok(generation)
}

/// Send a user message to the runtime: push display block, log, and spawn turn run.
#[allow(clippy::too_many_arguments)]
fn send_user_message(
    display_input: String,
    turn_input: TurnInput,
    app: &mut App,
    logger: &mut SessionLogger,
    runtime: &mut Option<LashRuntime>,
    _history: &mut Vec<Message>,
    runtime_return_rx: &mut Option<tokio::sync::oneshot::Receiver<RuntimeRunResult>>,
    cancel_token: &mut Option<CancellationToken>,
    active_stream_id: &mut u64,
    app_tx: &mpsc::UnboundedSender<AppEvent>,
) {
    let already_visible = if !display_input.is_empty() {
        app.commit_pending_user_preview(&display_input)
    } else {
        false
    };
    if !display_input.is_empty() && !already_visible {
        app.blocks
            .push(DisplayBlock::UserInput(display_input.clone()));
        app.invalidate_height_cache();
    }
    app.running = true;
    app.iteration = 0;
    app.resume_follow_output();
    app.keep_latest_user_block_visible();

    if !display_input.is_empty() {
        logger.log_user_input(&display_input);
    }

    let mut rt = runtime
        .take()
        .expect("runtime should be available when not running");
    tracing::info!(
        mode = ?turn_input.mode,
        items = turn_input.items.len(),
        images = turn_input.image_blobs.len(),
        "dispatching runtime turn"
    );
    let (return_tx, return_rx) = tokio::sync::oneshot::channel();
    *runtime_return_rx = Some(return_rx);

    let cancel = CancellationToken::new();
    *cancel_token = Some(cancel.clone());
    *active_stream_id = active_stream_id.wrapping_add(1);
    let stream_id = *active_stream_id;

    let sink_tx = app_tx.clone();
    tokio::spawn(async move {
        let sink = AppEventSink {
            tx: sink_tx,
            stream_id,
        };
        let result = match rt.stream_turn(turn_input, &sink, cancel).await {
            Ok(turn) => turn,
            Err(e) => AssembledTurn {
                state: rt.export_state(),
                status: TurnStatus::Failed,
                assistant_output: AssistantOutput {
                    safe_text: String::new(),
                    raw_text: String::new(),
                    state: OutputState::EmptyOutput,
                },
                done_reason: DoneReason::RuntimeError,
                execution: ExecutionSummary {
                    mode: rt.export_state().execution_mode,
                    had_tool_calls: false,
                    had_code_execution: false,
                },
                token_usage: TokenUsage::default(),
                tool_calls: Vec::new(),
                code_outputs: Vec::new(),
                errors: vec![TurnIssue {
                    kind: "runtime".to_string(),
                    code: Some(e.code),
                    message: e.message,
                }],
            },
        };
        let _ = return_tx.send(RuntimeRunResult {
            stream_id,
            runtime: rt,
            result,
        });
    });
}

/// Send a desktop notification that the agent finished.
fn notify_done() {
    // Ensure the icon exists in $LASH_HOME
    let icon_path = lash_core::lash_home().join("icon.svg");
    if !icon_path.exists() {
        let _ = std::fs::write(&icon_path, include_bytes!("../assets/icon.svg"));
    }
    let _ = std::process::Command::new("notify-send")
        .args(["-a", "lash", "-i"])
        .arg(&icon_path)
        .args(["lash", "Response complete"])
        .spawn();
}

/// Generate a unique session name like "juniper-mountain".
/// Scans existing session files for collisions.
fn generate_session_name(sessions_dir: &std::path::Path) -> String {
    use rand::Rng;
    use std::io::{BufRead, BufReader};

    const ADJECTIVES: &[&str] = &[
        "alpine", "amber", "ancient", "ashen", "autumn", "blazing", "bright", "calm", "cedar",
        "coastal", "copper", "coral", "crimson", "crystal", "dappled", "deep", "desert", "distant",
        "dusky", "ember", "fading", "fern", "flint", "foggy", "forest", "frozen", "gentle",
        "gilded", "glacial", "golden", "granite", "hollow", "iron", "ivory", "jade", "keen",
        "lofty", "lunar", "marble", "misty", "mossy", "northern", "obsidian", "onyx", "opal",
        "pale", "pine", "quiet", "radiant", "rugged", "rustic", "sandy", "silver", "silent",
        "solar", "stone", "sunlit", "tidal", "twilight", "verdant", "violet", "wild", "winter",
    ];
    const NOUNS: &[&str] = &[
        "basin",
        "birch",
        "bluff",
        "boulder",
        "brook",
        "canyon",
        "cavern",
        "cliff",
        "cove",
        "creek",
        "delta",
        "dune",
        "falls",
        "field",
        "fjord",
        "glade",
        "gorge",
        "grove",
        "harbor",
        "heath",
        "hill",
        "island",
        "lake",
        "ledge",
        "marsh",
        "meadow",
        "mesa",
        "mountain",
        "oasis",
        "ocean",
        "pass",
        "peak",
        "plain",
        "plateau",
        "pond",
        "prairie",
        "ravine",
        "reef",
        "ridge",
        "river",
        "shore",
        "slope",
        "spring",
        "stone",
        "summit",
        "terrace",
        "thicket",
        "timber",
        "trail",
        "tundra",
        "vale",
        "valley",
        "vista",
        "volcano",
        "waterfall",
        "willow",
        "woods",
    ];

    // Collect existing session names from session files
    let mut existing = std::collections::HashSet::new();
    if let Ok(entries) = std::fs::read_dir(sessions_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("jsonl")
                && let Ok(file) = std::fs::File::open(&path)
            {
                let reader = BufReader::new(file);
                if let Some(Ok(first_line)) = reader.lines().next()
                    && let Ok(v) = serde_json::from_str::<serde_json::Value>(&first_line)
                    && let Some(name) = v.get("session_name").and_then(|n| n.as_str())
                {
                    existing.insert(name.to_string());
                }
            }
        }
    }

    let mut rng = rand::rng();
    loop {
        let adj = ADJECTIVES[rng.random_range(0..ADJECTIVES.len())];
        let noun = NOUNS[rng.random_range(0..NOUNS.len())];
        let name = format!("{}-{}", adj, noun);
        if !existing.contains(&name) {
            return name;
        }
    }
}

/// Copy the last assistant response to the system clipboard.
fn copy_last_response(app: &App) {
    let last_text = app.blocks.iter().rev().find_map(|b| {
        if let DisplayBlock::AssistantText(text) = b {
            Some(text.clone())
        } else {
            None
        }
    });
    if let Some(text) = last_text
        && let Ok(mut clipboard) = arboard::Clipboard::new()
    {
        let _ = clipboard.set_text(text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_inline_marker_adds_spaces_when_touching_text() {
        let mut app = App::new("model".into(), "session".into());
        app.input = "hello world".into();
        app.cursor_pos = 5;
        insert_inline_marker(&mut app, "[Image #1]");
        assert_eq!(app.input, "hello [Image #1] world");
    }

    #[test]
    fn insert_inline_marker_keeps_existing_spacing() {
        let mut app = App::new("model".into(), "session".into());
        app.input = "hello ".into();
        app.cursor_pos = app.input.len();
        insert_inline_marker(&mut app, "[Image #1]");
        assert_eq!(app.input, "hello [Image #1]");
    }

    #[test]
    fn parse_image_marker_rejects_zero_index() {
        assert_eq!(input_items::parse_image_marker_at("[Image #0]", 0), None);
    }

    #[test]
    fn build_items_preserves_interleaving_for_images_and_paths() {
        let unique = format!(
            "lash-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        );
        let tmp_path = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&tmp_path).expect("mkdir temp test dir");
        let file_path = tmp_path.join("a.txt");
        std::fs::write(&file_path, "x").expect("write temp file");
        let original_cwd = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&tmp_path).expect("chdir");
        let (items, image_blobs) =
            build_items_from_editor_input("before [Image #1] @a.txt after", vec![vec![1, 2, 3]]);
        std::env::set_current_dir(original_cwd).expect("restore cwd");
        let _ = std::fs::remove_dir_all(&tmp_path);

        let kinds: Vec<&'static str> = items
            .iter()
            .map(|item| match item {
                InputItem::Text { .. } => "text",
                InputItem::ImageRef { .. } => "image",
                InputItem::FileRef { .. } => "file",
                InputItem::DirRef { .. } => "dir",
                InputItem::SkillRef { .. } => "skill",
            })
            .collect();
        assert_eq!(kinds, vec!["text", "image", "text", "file", "text"]);
        assert_eq!(image_blobs.len(), 1);
    }

    #[test]
    fn controls_text_mentions_alt_up_queue_edit_binding() {
        let controls = controls_text();
        assert!(controls.contains("Alt+Up             Edit last queued turn"));
        assert!(!controls.contains("Backspace          Restore last next-turn draft"));
    }
}
