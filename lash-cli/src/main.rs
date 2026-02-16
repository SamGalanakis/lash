mod app;
mod command;
mod event;
mod markdown;
#[allow(dead_code)]
mod session_log;
mod setup;
mod skill;
#[allow(dead_code)]
mod theme;
mod ui;
mod util;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::Parser;
use crossterm::cursor::SetCursorStyle;
use crossterm::event::{Event as TermEvent, KeyCode, KeyEventKind, KeyModifiers};
use lash::agent::{Message, MessageRole, Part, PartKind, PruneState};
use lash::provider::LashConfig;
use lash::tools::{
    CompositeTools, DelegateDeep, DelegateSearch, DelegateTask, DiffFile, EditFile, FetchUrl,
    FindReplace, Glob, Grep, Ls, ReadFile, Shell, SkillStore, TaskStore, ViewMessage, WebSearch,
    WriteFile,
};
use lash::*;
use ratatui::DefaultTerminal;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use app::{App, DisplayBlock};
use event::AppEvent;

#[derive(Parser)]
struct Args {
    /// OpenRouter API key (optional — use --provider to configure interactively)
    #[arg(long, env = "OPENROUTER_API_KEY")]
    api_key: Option<String>,

    /// Tavily API key for web search
    #[arg(long, env = "TAVILY_API_KEY")]
    tavily_api_key: Option<String>,

    /// Model name (defaults per provider: claude-opus-4-6 for Claude, anthropic/claude-opus-4.6 for OpenRouter)
    #[arg(long)]
    model: Option<String>,

    /// Base URL for the LLM API
    #[arg(long, default_value = "https://openrouter.ai/api/v1")]
    base_url: String,

    /// Disable mouse scroll support (re-enables terminal text selection)
    #[arg(long)]
    no_mouse: bool,

    /// Force re-run provider setup
    #[arg(long)]
    provider: bool,

    /// Delete all lash data (~/.lash/ and ~/.cache/lash/) and exit
    #[arg(long)]
    reset: bool,

    /// Run headlessly: execute prompt, print response to stdout, exit
    #[arg(short = 'p', long = "print")]
    print_prompt: Option<String>,
}

#[allow(dead_code)]
struct SessionLogger {
    file: std::fs::File,
    session_id: String,
    session_name: String,
}

impl SessionLogger {
    fn new(model: &str) -> anyhow::Result<Self> {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let dir = PathBuf::from(home).join(".lash").join("sessions");
        std::fs::create_dir_all(&dir)?;

        let now = chrono::Local::now();
        let filename = format!("{}.jsonl", now.format("%Y%m%d_%H%M%S"));
        let path = dir.join(&filename);
        let file = std::fs::File::create(&path)?;
        let session_id = uuid::Uuid::new_v4().to_string();
        let session_name = generate_session_name(&dir);

        let mut logger = Self {
            file,
            session_id: session_id.clone(),
            session_name: session_name.clone(),
        };
        logger.write_json(&serde_json::json!({
            "type": "session_start",
            "session_id": session_id,
            "session_name": session_name,
            "ts": now.to_rfc3339(),
            "model": model,
            "cwd": std::env::current_dir().ok().map(|p| p.to_string_lossy().to_string()),
        }))?;

        Ok(logger)
    }

    fn write_json(&mut self, value: &serde_json::Value) -> anyhow::Result<()> {
        use std::io::Write;
        serde_json::to_writer(&mut self.file, value)?;
        self.file.write_all(b"\n")?;
        self.file.flush()?;
        Ok(())
    }

    fn log_user_input(&mut self, input: &str) {
        let _ = self.write_json(&serde_json::json!({
            "type": "user_input",
            "ts": chrono::Local::now().to_rfc3339(),
            "content": input,
        }));
    }

    fn log_event(&mut self, event: &AgentEvent) {
        let mut value = serde_json::to_value(event).unwrap_or_default();
        if let serde_json::Value::Object(ref mut map) = value {
            map.insert(
                "ts".into(),
                serde_json::Value::String(chrono::Local::now().to_rfc3339()),
            );
        }
        let _ = self.write_json(&value);
    }
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Suppress BAML's prompt/response logging — it writes to stderr which breaks the TUI
    if std::env::var("BAML_LOG").is_err() {
        // SAFETY: called before any threads are spawned
        unsafe { std::env::set_var("BAML_LOG", "off") };
    }

    // Set up file-based tracing (logs go to ~/.lash/lash.log)
    {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let log_dir = PathBuf::from(&home).join(".lash");
        std::fs::create_dir_all(&log_dir).ok();
        let log_file = std::fs::File::create(log_dir.join("lash.log"))?;

        use tracing_subscriber::EnvFilter;
        let filter = EnvFilter::try_from_env("LASH_LOG").unwrap_or_else(|_| EnvFilter::new("warn"));
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(log_file)
            .with_ansi(false)
            .init();
    }

    let args = Args::parse();

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

        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let lash_dir = PathBuf::from(&home).join(".lash");
        let cache_dir = PathBuf::from(&home).join(".cache").join("lash");

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
            "    {ASH_TEXT}python cache           {CHALK}{}{RESET}",
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
    let mut lash_config = if args.provider || LashConfig::load().is_none() {
        if let Some(ref key) = args.api_key {
            // Shortcut: env var or --api-key creates OpenRouter provider directly
            LashConfig {
                provider: Provider::OpenRouter {
                    api_key: key.clone(),
                    base_url: args.base_url.clone(),
                },
                tavily_api_key: args.tavily_api_key.clone(),
                delegate_models: None,
            }
        } else if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            // ANTHROPIC_API_KEY env var → use as direct Claude bearer token
            LashConfig {
                provider: Provider::Claude {
                    access_token: key,
                    refresh_token: String::new(),
                    expires_at: u64::MAX,
                },
                tavily_api_key: args.tavily_api_key.clone(),
                delegate_models: None,
            }
        } else {
            setup::run_setup().await?
        }
    } else {
        let mut c = LashConfig::load().unwrap();
        if c.provider.ensure_fresh().await? {
            c.save()?; // persist refreshed tokens
        }
        c
    };

    // CLI env/flags override stored config
    if let Some(ref key) = args.tavily_api_key {
        lash_config.tavily_api_key = Some(key.clone());
    }
    if args.print_prompt.is_none() {
        lash_config.save()?;
    }

    let model = args
        .model
        .clone()
        .unwrap_or_else(|| lash_config.provider.default_model().to_string());

    let llm_log_path = std::env::var("LASH_LOG").ok().and_then(|level| {
        let l = level.to_lowercase();
        if l == "debug" || l == "trace" || l.contains("debug") || l.contains("trace") {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            let dir = PathBuf::from(home).join(".lash").join("sessions");
            Some(dir.join(format!(
                "{}.llm.jsonl",
                chrono::Local::now().format("%Y%m%d_%H%M%S")
            )))
        } else {
            None
        }
    });

    let config = AgentConfig {
        model: model.clone(),
        provider: lash_config.provider.clone(),
        llm_log_path,
        headless: args.print_prompt.is_some(),
        ..Default::default()
    };

    // Build store (SQLite-backed archive + tasks)
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let sessions_dir = PathBuf::from(&home).join(".lash").join("sessions");
    std::fs::create_dir_all(&sessions_dir)?;
    let db_path = sessions_dir.join(format!(
        "{}.db",
        chrono::Local::now().format("%Y%m%d_%H%M%S")
    ));
    let store = Arc::new(Store::open(&db_path).expect("Failed to open session database"));

    let task_store: Arc<TaskStore> = Arc::new(TaskStore::new(Arc::clone(&store)));
    let instruction_loader = Arc::new(lash::InstructionLoader::new());

    let mut base = CompositeTools::new()
        .add(Shell::new())
        .add(ReadFile::new(Arc::clone(&instruction_loader)))
        .add(WriteFile)
        .add(EditFile)
        .add(DiffFile)
        .add(FindReplace)
        .add(Glob)
        .add(Grep)
        .add(Ls)
        .add(ViewMessage::new(Arc::clone(&store)))
        .add_arc(Arc::clone(&task_store) as Arc<dyn ToolProvider>);
    if let Some(ref key) = lash_config.tavily_api_key {
        base = base.add(WebSearch::new(key)).add(FetchUrl::new(key));
    }
    let base_tools: Arc<dyn ToolProvider> = Arc::new(base);

    // SkillStore — available to root + balanced/deep delegates, NOT search delegates
    let skill_dirs = vec![
        PathBuf::from(&home).join(".lash").join("skills"),
        PathBuf::from(".lash").join("skills"),
    ];
    let tools_with_skills: Arc<dyn ToolProvider> = Arc::new(
        CompositeTools::new()
            .add_arc(Arc::clone(&base_tools))
            .add(SkillStore::new(skill_dirs)),
    );

    // Root cancel token — lives for the whole app lifetime, child tokens are created per run
    let root_cancel = CancellationToken::new();

    let tools: Arc<dyn ToolProvider> = Arc::new(
        CompositeTools::new()
            .add_arc(Arc::clone(&tools_with_skills))
            .add(DelegateSearch::new(
                Arc::clone(&base_tools),
                &config,
                lash_config.delegate_models.clone(),
                Arc::clone(&store),
                root_cancel.clone(),
                "root".to_string(),
            ))
            .add(DelegateTask::new(
                Arc::clone(&tools_with_skills),
                &config,
                lash_config.delegate_models.clone(),
                Arc::clone(&store),
                root_cancel.clone(),
                "root".to_string(),
            ))
            .add(DelegateDeep::new(
                Arc::clone(&tools_with_skills),
                &config,
                lash_config.delegate_models.clone(),
                Arc::clone(&store),
                root_cancel.clone(),
                "root".to_string(),
            )),
    );
    let session = Session::new(tools, SessionConfig::default(), "root").await?;

    let agent = Agent::new(
        session,
        config,
        Arc::clone(&store),
        Some("root".to_string()),
    );

    // ── Headless mode: skip TUI, run agent, print to stdout ──
    if let Some(prompt) = args.print_prompt {
        return run_headless(agent, prompt, Arc::clone(&store)).await;
    }

    let mut logger = SessionLogger::new(&model)?;

    // Install panic hook that restores the terminal
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        cleanup_terminal();
        default_hook(info);
    }));

    // Initialize terminal
    let terminal = ratatui::init();

    // Set terminal background to match FORM so padding areas blend seamlessly (OSC 11)
    // Set cursor to bar (pipe) style for cleaner text editing feel
    crossterm::execute!(
        std::io::stdout(),
        crossterm::style::Print("\x1b]11;rgb:0e/0d/0b\x1b\\"),
        SetCursorStyle::SteadyBar
    )?;

    // Enable kitty keyboard protocol so Shift+Enter is distinguishable from Enter
    let _ = crossterm::execute!(
        std::io::stdout(),
        crossterm::event::PushKeyboardEnhancementFlags(
            crossterm::event::KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
        )
    );

    // Enable mouse capture by default (scroll wheel works always)
    if !args.no_mouse {
        crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture)?;
    }

    // Enable focus change tracking for gating desktop notifications
    crossterm::execute!(std::io::stdout(), crossterm::event::EnableFocusChange)?;

    let session_name = logger.session_name.clone();
    let result = run_app(
        terminal,
        agent,
        &mut logger,
        &args,
        model,
        session_name,
        Arc::clone(&store),
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

/// Run the agent headlessly: send prompt, consume events, print final response to stdout.
async fn run_headless(mut agent: Agent, prompt: String, _store: Arc<Store>) -> anyhow::Result<()> {
    let prompt = transform_at_references(&prompt);

    let history = vec![Message {
        id: "m0".to_string(),
        role: MessageRole::User,
        parts: vec![Part {
            id: "m0.p0".to_string(),
            kind: PartKind::Text,
            content: prompt,
            prune_state: PruneState::Intact,
        }],
    }];

    let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(100);
    let cancel = CancellationToken::new();

    // Spawn agent run
    let (return_tx, return_rx) = tokio::sync::oneshot::channel::<AgentRunResult>();
    let msgs = history.clone();
    tokio::spawn(async move {
        let (new_history, final_turn) = agent.run(msgs, Vec::new(), event_tx, cancel, 0).await;
        let _ = return_tx.send(AgentRunResult {
            agent,
            history: new_history,
            turn: final_turn,
        });
    });

    let mut exit_code = 0i32;

    // Consume events
    while let Some(event) = event_rx.recv().await {
        match event {
            AgentEvent::Message { text, kind } => {
                if kind == "final" {
                    println!("{}", text);
                }
            }
            AgentEvent::ToolCall { .. } => {}
            AgentEvent::Error { message } => {
                eprintln!("error: {}", message);
                exit_code = 1;
            }
            AgentEvent::Prompt { response_tx, .. } => {
                // No human available — send empty string
                let _ = response_tx.send(String::new());
            }
            AgentEvent::Done => break,
            // Ignore streaming deltas, code blocks, token usage, etc.
            _ => {}
        }
    }

    // Wait for agent task to finish
    let _ = return_rx.await;

    if exit_code != 0 {
        std::process::exit(exit_code);
    }
    Ok(())
}

/// Returned by the spawned agent task so we can reclaim ownership.
struct AgentRunResult {
    agent: Agent,
    history: Vec<Message>,
    turn: usize,
}

/// Build the controls text shown by /controls.
fn controls_text() -> String {
    [
        "Controls:",
        "  Shift+Tab          Toggle plan mode",
        "  Esc                Cancel agent (while running)",
        "  Enter              Queue message (while running)",
        "  Backspace          Unqueue last (while running)",
        "  Ctrl+U / Ctrl+D    Scroll half-page up / down",
        "  PgUp / PgDn        Scroll page up / down",
        "  Shift+Enter        Insert newline",
        "  Ctrl+V             Paste image (or text fallback)",
        "  Ctrl+Shift+V       Paste text only",
        "  Ctrl+Y             Copy last response to clipboard",
        "  Ctrl+O             Toggle code block expansion",
        "  Up / Down          Input history",
        "  Shift+Drag         Select text (terminal native)",
        "  Ctrl+C             Quit",
    ]
    .join("\n")
}

/// Build the help text shown by /help.
fn help_text(skills: &skill::SkillRegistry) -> String {
    let mut lines = vec![
        "Commands:".to_string(),
        "  /clear, /new       Reset conversation".to_string(),
        "  /model <name>      Switch LLM model".to_string(),
        "  /provider          Change LLM provider".to_string(),
        "  /logout            Remove stored credentials".to_string(),
        "  /resume [name]     Browse or load a previous session".to_string(),
        "  /skills            Browse loaded skills".to_string(),
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
        "Shortcuts:".to_string(),
        "  Shift+Tab          Toggle plan mode".to_string(),
        "  Esc                Cancel agent (while running)".to_string(),
        "  Ctrl+U / Ctrl+D    Scroll half-page up / down".to_string(),
        "  PgUp / PgDn        Scroll page up / down".to_string(),
        "  Shift+Enter        Insert newline".to_string(),
        "  Ctrl+V             Paste image (or text fallback)".to_string(),
        "  Ctrl+Shift+V       Paste text only".to_string(),
        "  Ctrl+Y             Copy last response to clipboard".to_string(),
        "  Ctrl+O             Toggle code block expansion".to_string(),
        "  Shift+Drag         Select text (terminal native)".to_string(),
        "  Up/Down            Input history".to_string(),
        "  Ctrl+C             Quit".to_string(),
    ]);

    lines.join("\n")
}

async fn run_app(
    mut terminal: DefaultTerminal,
    agent: Agent,
    logger: &mut SessionLogger,
    _args: &Args,
    model: String,
    session_name: String,
    store: Arc<Store>,
) -> anyhow::Result<()> {
    let mut app = App::new(model, session_name, Some(store));
    app.load_history();
    let mut history: Vec<Message> = Vec::new();
    let mut turn_counter: usize = 0;
    let mut agent = Some(agent);

    // Cancellation token for interrupting a running agent
    let mut cancel_token: Option<CancellationToken> = None;

    // Unified event channel
    let (app_tx, mut app_rx) = mpsc::unbounded_channel::<AppEvent>();

    // Stop flag for the event reader thread
    let stop = Arc::new(AtomicBool::new(false));

    // Spawn terminal event reader using poll() with timeout so it can stop
    let term_tx = app_tx.clone();
    let stop_reader = Arc::clone(&stop);
    tokio::task::spawn_blocking(move || {
        while !stop_reader.load(Ordering::Relaxed) {
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
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(100));
        loop {
            interval.tick().await;
            if stop_tick.load(Ordering::Relaxed) {
                break;
            }
            if tick_tx.send(AppEvent::Tick).is_err() {
                break;
            }
        }
    });

    // SIGTERM handler for graceful shutdown
    let sigterm_tx = app_tx.clone();
    tokio::spawn(async move {
        use tokio::signal::unix::{SignalKind, signal};
        if let Ok(mut sig) = signal(SignalKind::terminate()) {
            sig.recv().await;
            let _ = sigterm_tx.send(AppEvent::Quit);
        }
    });

    // Oneshot for receiving agent back after a run completes
    let mut agent_return_rx: Option<tokio::sync::oneshot::Receiver<AgentRunResult>> = None;

    loop {
        // Check if agent run completed — reclaim agent + updated history
        if let Some(ref mut rx) = agent_return_rx {
            match rx.try_recv() {
                Ok(result) => {
                    agent = Some(result.agent);
                    history = result.history;
                    turn_counter = result.turn;
                    agent_return_rx = None;
                    cancel_token = None;

                    // Auto-drain: if there are queued messages, send the next one
                    if let Some(queued_input) = app.dequeue_message() {
                        send_user_message(
                            queued_input,
                            Vec::new(),
                            &mut app,
                            logger,
                            &mut history,
                            &mut agent,
                            &mut agent_return_rx,
                            &mut cancel_token,
                            &app_tx,
                            turn_counter,
                        );
                    }
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    app.running = false;
                    agent_return_rx = None;
                    cancel_token = None;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {}
            }
        }

        // Draw only when dirty
        if app.dirty {
            // Pre-compute height cache before immutable borrow in draw
            let size = terminal.size()?;
            let overhead: u16 =
                (if app.running { 5 } else { 4 }) + app.task_tray_height(size.width);
            let vh = size.height.saturating_sub(overhead) as usize;
            let vw = size.width as usize;
            app.ensure_height_cache_pub(vw, vh);
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
                app.dirty = true;
                // CTRL+C: dismiss prompt if active, else quit
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                    if app.has_prompt() {
                        app.dismiss_prompt();
                        continue;
                    }
                    break;
                }

                // CTRL+O: toggle code expand (works in all modes)
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('o') {
                    app.toggle_code_expand();
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
                    let overhead: u16 =
                        (if app.running { 5 } else { 4 }) + app.task_tray_height(size.width);
                    let vh = size.height.saturating_sub(overhead) as usize;
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
                                match session_log::load_session(&filename) {
                                    Some((msgs, blocks)) => {
                                        history = msgs;
                                        app.blocks = blocks;
                                        app.blocks.push(DisplayBlock::SystemMessage(format!(
                                            "Resumed: {}",
                                            filename
                                        )));

                                        // Try to restore agent state from .db
                                        restore_agent_state(
                                            &filename,
                                            &mut history,
                                            &mut agent,
                                            &mut app,
                                        )
                                        .await;

                                        app.invalidate_height_cache();
                                        app.scroll_to_bottom();
                                    }
                                    None => {
                                        app.blocks.push(DisplayBlock::SystemMessage(format!(
                                            "Could not load: {}",
                                            filename
                                        )));
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

                // Shift+Tab: toggle plan mode (not in prompt/picker)
                if key.code == KeyCode::BackTab {
                    app.mode = match app.mode {
                        app::Mode::Normal => app::Mode::Plan,
                        app::Mode::Plan => app::Mode::Normal,
                    };
                    let mode_label = match app.mode {
                        app::Mode::Normal => "Switched to normal mode".to_string(),
                        app::Mode::Plan => {
                            let path = app.ensure_plan_file();
                            // Show relative path if possible
                            let display = std::env::current_dir()
                                .ok()
                                .and_then(|cwd| {
                                    path.strip_prefix(&cwd).ok().map(|p| p.to_path_buf())
                                })
                                .unwrap_or_else(|| path.clone());
                            format!("Switched to plan mode \u{2014} {}", display.display())
                        }
                    };
                    app.blocks.push(DisplayBlock::SystemMessage(mode_label));
                    app.invalidate_height_cache();
                    app.scroll_to_bottom();
                    continue;
                }

                match key.code {
                    // Tab: complete selected suggestion
                    KeyCode::Tab if app.has_suggestions() => {
                        app.complete_suggestion();
                        app.update_suggestions();
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
                        app.update_suggestions();
                        if input.is_empty() && app.pending_images.is_empty() {
                            continue;
                        }

                        if app.running {
                            // Queue for later — skip slash commands, just buffer
                            app.queue_message(input);
                            continue;
                        }

                        // Shell escape: !command
                        if let Some(cmd_str) = input.strip_prefix('!') {
                            let cmd_str = cmd_str.trim();
                            if !cmd_str.is_empty() {
                                app.blocks.push(DisplayBlock::UserInput(input.clone()));
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
                                            "Command '{}' timed out after 30s",
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
                        if let Some(cmd) = command::parse(&input, &app.skills) {
                            match cmd {
                                command::Command::Exit => break,
                                command::Command::Clear => {
                                    app.clear();
                                    history.clear();
                                    if let Some(ag) = agent.as_mut() {
                                        let _ = ag.reset_session().await;
                                    }
                                }
                                command::Command::Model(new_model) => {
                                    if let Some(ag) = agent.as_mut() {
                                        ag.set_model(new_model.clone());
                                    }
                                    app.context_window =
                                        lash::model_info::context_window(&new_model);
                                    app.model = new_model;
                                }
                                command::Command::ChangeProvider => {
                                    app.blocks.push(DisplayBlock::SystemMessage(
                                        "Restart lash with `--provider` to change providers."
                                            .to_string(),
                                    ));
                                    app.invalidate_height_cache();
                                    app.scroll_to_bottom();
                                }
                                command::Command::Logout => {
                                    match LashConfig::clear() {
                                        Ok(()) => {
                                            app.blocks.push(DisplayBlock::SystemMessage(
                                                "Credentials removed.".to_string(),
                                            ));
                                        }
                                        Err(e) => {
                                            app.blocks.push(DisplayBlock::SystemMessage(format!(
                                                "Failed to remove credentials: {}",
                                                e
                                            )));
                                        }
                                    }
                                    app.invalidate_height_cache();
                                    app.scroll_to_bottom();
                                }
                                command::Command::Controls => {
                                    app.blocks
                                        .push(DisplayBlock::SystemMessage(controls_text()));
                                    app.invalidate_height_cache();
                                    app.scroll_to_bottom();
                                }
                                command::Command::Help => {
                                    app.blocks
                                        .push(DisplayBlock::SystemMessage(help_text(&app.skills)));
                                    app.invalidate_height_cache();
                                    app.scroll_to_bottom();
                                }
                                command::Command::Resume(name) => {
                                    if let Some(filename) = name {
                                        // Direct load by filename
                                        match session_log::load_session(&filename) {
                                            Some((msgs, blocks)) => {
                                                history = msgs;
                                                app.blocks = blocks;
                                                app.blocks.push(DisplayBlock::SystemMessage(
                                                    format!("Resumed: {}", filename),
                                                ));

                                                // Try to restore agent state from .db
                                                restore_agent_state(
                                                    &filename,
                                                    &mut history,
                                                    &mut agent,
                                                    &mut app,
                                                )
                                                .await;

                                                app.invalidate_height_cache();
                                                app.scroll_to_bottom();
                                            }
                                            None => {
                                                app.blocks.push(DisplayBlock::SystemMessage(
                                                    format!("Could not load: {}", filename),
                                                ));
                                                app.invalidate_height_cache();
                                                app.scroll_to_bottom();
                                            }
                                        }
                                    } else {
                                        // No arg — open session picker
                                        let mut sessions = session_log::list_sessions();
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
                                command::Command::Skills => {
                                    let items: Vec<(String, String)> = app
                                        .skills
                                        .iter()
                                        .map(|s| (s.name.clone(), s.description.clone()))
                                        .collect();
                                    if items.is_empty() {
                                        app.blocks.push(DisplayBlock::SystemMessage(
                                            "No skills found.\n\
                                             Add skill directories to ~/.lash/skills/ or .lash/skills/\n\
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
                                        let user_msg = match args {
                                            Some(a) => format!("[SKILL:{}] {}", name, a),
                                            None => format!("[SKILL:{}]", name),
                                        };

                                        // Reuse send_user_message — display shows original input,
                                        // but we send [SKILL:name] as the user message content
                                        app.blocks.push(DisplayBlock::UserInput(input.clone()));
                                        app.invalidate_height_cache();
                                        app.scroll_to_bottom();
                                        app.running = true;
                                        app.iteration = 0;

                                        logger.log_user_input(&input);

                                        let usr_id = format!("m{}", history.len());
                                        history.push(Message {
                                            id: usr_id.clone(),
                                            role: MessageRole::User,
                                            parts: vec![Part {
                                                id: format!("{}.p0", usr_id),
                                                kind: PartKind::Text,
                                                content: user_msg,
                                                prune_state: PruneState::Intact,
                                            }],
                                        });

                                        let (event_tx, mut event_rx) =
                                            mpsc::channel::<AgentEvent>(100);
                                        let fwd_tx = app_tx.clone();
                                        tokio::spawn(async move {
                                            while let Some(ev) = event_rx.recv().await {
                                                if fwd_tx.send(AppEvent::Agent(ev)).is_err() {
                                                    break;
                                                }
                                            }
                                        });

                                        let mut ag = agent
                                            .take()
                                            .expect("agent should be available when not running");
                                        let msgs = history.clone();
                                        let (return_tx, return_rx) =
                                            tokio::sync::oneshot::channel();
                                        agent_return_rx = Some(return_rx);

                                        let cancel = CancellationToken::new();
                                        cancel_token = Some(cancel.clone());

                                        let offset = turn_counter;
                                        tokio::spawn(async move {
                                            let (new_history, final_turn) = ag
                                                .run(msgs, Vec::new(), event_tx, cancel, offset)
                                                .await;
                                            let _ = return_tx.send(AgentRunResult {
                                                agent: ag,
                                                history: new_history,
                                                turn: final_turn,
                                            });
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
                        let images = app.take_images();
                        send_user_message(
                            input,
                            images,
                            &mut app,
                            logger,
                            &mut history,
                            &mut agent,
                            &mut agent_return_rx,
                            &mut cancel_token,
                            &app_tx,
                            turn_counter,
                        );
                    }
                    KeyCode::Backspace => {
                        if app.input.is_empty() && app.queue_count() > 0 {
                            // Pop last queued message back to editor
                            if let Some(msg) = app.unqueue_last() {
                                app.input = msg;
                                app.cursor_pos = app.input.len();
                            }
                        } else {
                            app.backspace();
                        }
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
                app.dirty = true;
                use crossterm::event::MouseEventKind;
                match mouse.kind {
                    MouseEventKind::ScrollUp => app.scroll_up(3),
                    MouseEventKind::ScrollDown => {
                        let size = terminal.size()?;
                        let overhead: u16 =
                            (if app.running { 5 } else { 4 }) + app.task_tray_height(size.width);
                        let vh = size.height.saturating_sub(overhead) as usize;
                        let vw = size.width as usize;
                        app.scroll_down(3, vh, vw);
                    }
                    _ => {}
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
                // Auto-dismiss task tray after countdown
                if app.task_all_done_at.is_some() {
                    app.dirty = true; // keep redrawing for countdown
                    app.maybe_dismiss_task_tray();
                }
            }
            AppEvent::Agent(event) => {
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
                    if is_done {
                        // Display plan content if plan file was modified
                        if app.mode == app::Mode::Plan
                            && let Some(ref plan_path) = app.plan_file
                        {
                            let new_mtime = std::fs::metadata(plan_path)
                                .ok()
                                .and_then(|m| m.modified().ok());
                            if new_mtime != app.plan_file_mtime && plan_path.exists() {
                                app.plan_file_mtime = new_mtime;
                                if let Ok(content) = std::fs::read_to_string(plan_path) {
                                    app.blocks.push(DisplayBlock::PlanContent(content));
                                    app.invalidate_height_cache();
                                    app.scroll_to_bottom();
                                }
                            }
                        }
                        if !app.focused {
                            notify_done();
                        }
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

    Ok(())
}

/// Send a user message to the agent: push display block, transform refs, log, update history, spawn agent run.
#[allow(clippy::too_many_arguments)]
fn send_user_message(
    input: String,
    images: Vec<Vec<u8>>,
    app: &mut App,
    logger: &mut SessionLogger,
    history: &mut Vec<Message>,
    agent: &mut Option<Agent>,
    agent_return_rx: &mut Option<tokio::sync::oneshot::Receiver<AgentRunResult>>,
    cancel_token: &mut Option<CancellationToken>,
    app_tx: &mpsc::UnboundedSender<AppEvent>,
    turn_counter: usize,
) {
    app.blocks.push(DisplayBlock::UserInput(input.clone()));
    app.invalidate_height_cache();
    app.scroll_to_bottom();
    app.running = true;
    app.iteration = 0;

    let input = transform_at_references(&input);

    logger.log_user_input(&input);

    // Inject plan mode system message before the user message
    if app.mode == app::Mode::Plan {
        let plan_path = app.ensure_plan_file();
        let sys_id = format!("m{}", history.len());
        history.push(Message {
            id: sys_id.clone(),
            role: MessageRole::System,
            parts: vec![Part {
                id: format!("{}.p0", sys_id),
                kind: PartKind::Text,
                content: format!(
                    "## Plan Mode\n\n\
                    You are in PLAN mode. Think, explore, and design \u{2014} do NOT execute changes.\n\n\
                    **Rules:**\n\
                    - READ-ONLY: Do not modify project files or run destructive commands\n\
                    - You MAY use: read_file, glob, grep, ls, web_search, fetch_url, delegate_task\n\
                    - You MUST NOT use: edit_file, find_replace, write_file (except the plan file), or shell with write commands\n\
                    - Exception: Write your plan to `{}` using write_file\n\n\
                    **Workflow:**\n\
                    1. Understand the request \u{2014} ask clarifying questions using message(kind=\"final\")\n\
                    2. Explore the codebase \u{2014} read files, search for patterns, understand architecture\n\
                    3. Design your approach \u{2014} consider tradeoffs, identify risks\n\
                    4. Write a clear, step-by-step plan to the plan file\n\n\
                    When the user switches back to normal mode, you will execute the plan.",
                    plan_path.display()
                ),
                prune_state: PruneState::Intact,
            }],
        });
    } else if app.mode == app::Mode::Normal && app.plan_file.is_some() {
        let sys_id = format!("m{}", history.len());
        history.push(Message {
            id: sys_id.clone(),
            role: MessageRole::System,
            parts: vec![Part {
                id: format!("{}.p0", sys_id),
                kind: PartKind::Text,
                content: format!(
                    "You are back in normal mode. You may now execute changes.\n\
                    Plan file: {}",
                    app.plan_file.as_ref().unwrap().display()
                ),
                prune_state: PruneState::Intact,
            }],
        });
    }

    let usr_id = format!("m{}", history.len());
    history.push(Message {
        id: usr_id.clone(),
        role: MessageRole::User,
        parts: vec![Part {
            id: format!("{}.p0", usr_id),
            kind: PartKind::Text,
            content: input,
            prune_state: PruneState::Intact,
        }],
    });

    let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(100);
    let fwd_tx = app_tx.clone();
    tokio::spawn(async move {
        while let Some(ev) = event_rx.recv().await {
            if fwd_tx.send(AppEvent::Agent(ev)).is_err() {
                break;
            }
        }
    });

    let mut ag = agent
        .take()
        .expect("agent should be available when not running");
    let msgs = history.clone();
    let (return_tx, return_rx) = tokio::sync::oneshot::channel();
    *agent_return_rx = Some(return_rx);

    let cancel = CancellationToken::new();
    *cancel_token = Some(cancel.clone());

    let offset = turn_counter;
    tokio::spawn(async move {
        let (new_history, final_turn) = ag.run(msgs, images, event_tx, cancel, offset).await;
        let _ = return_tx.send(AgentRunResult {
            agent: ag,
            history: new_history,
            turn: final_turn,
        });
    });
}

/// Try to restore agent state from the .db file corresponding to a .jsonl session file.
/// On success, restores the REPL via dill. On failure (or no .db), injects a reset message.
async fn restore_agent_state(
    jsonl_filename: &str,
    history: &mut Vec<Message>,
    agent: &mut Option<Agent>,
    app: &mut App,
) {
    // Derive .db path from .jsonl filename (same stem)
    let stem = jsonl_filename.trim_end_matches(".jsonl");
    let db_filename = format!("{}.db", stem);
    let db_path = session_log::sessions_dir().join(&db_filename);

    if !db_path.exists() {
        // No .db file — inject reset message
        let sys_id = format!("m{}", history.len());
        history.push(Message {
            id: sys_id.clone(),
            role: MessageRole::System,
            parts: vec![Part {
                id: format!("{}.p0", sys_id),
                kind: PartKind::Text,
                content: "Session resumed. Your REPL environment was reset — re-import modules and recreate any state you need.".to_string(),
                prune_state: PruneState::Intact,
            }],
        });
        return;
    }

    // Open the .db and try to load root agent state
    let resume_store = match Store::open(&db_path) {
        Ok(s) => s,
        Err(_) => {
            app.blocks.push(DisplayBlock::SystemMessage(
                "Could not open session database.".to_string(),
            ));
            return;
        }
    };

    if let Some(state) = resume_store.load_agent_state("root") {
        // Restore token counts from DB
        if state.input_tokens > 0 || state.output_tokens > 0 {
            app.token_usage = lash::TokenUsage {
                input_tokens: state.input_tokens,
                output_tokens: state.output_tokens,
                cached_input_tokens: state.cached_input_tokens,
            };
        }

        if let Some(ref dill_blob) = state.dill_blob {
            // Try to restore REPL state
            if let Some(ag) = agent.as_mut() {
                match ag.restore(dill_blob).await {
                    Ok(()) => {
                        app.blocks.push(DisplayBlock::SystemMessage(
                            "REPL state restored from snapshot.".to_string(),
                        ));
                    }
                    Err(e) => {
                        let sys_id = format!("m{}", history.len());
                        history.push(Message {
                            id: sys_id.clone(),
                            role: MessageRole::System,
                            parts: vec![Part {
                                id: format!("{}.p0", sys_id),
                                kind: PartKind::Text,
                                content: format!(
                                    "Session resumed but REPL restore failed ({}). Re-import modules and recreate any state you need.",
                                    e
                                ),
                                prune_state: PruneState::Intact,
                            }],
                        });
                    }
                }
            }
        } else {
            // No dill blob — inject reset message
            let sys_id = format!("m{}", history.len());
            history.push(Message {
                id: sys_id.clone(),
                role: MessageRole::System,
                parts: vec![Part {
                    id: format!("{}.p0", sys_id),
                    kind: PartKind::Text,
                    content: "Session resumed. Your REPL environment was reset — re-import modules and recreate any state you need.".to_string(),
                    prune_state: PruneState::Intact,
                }],
            });
        }

        // Handle active sub-agents: inject context into parent history and mark them done
        let active_subs = resume_store.list_active_agents(Some("root"));
        for sub in &active_subs {
            // Extract prompt from config_json if available
            let prompt = serde_json::from_str::<serde_json::Value>(&sub.config_json)
                .ok()
                .and_then(|v| v.get("prompt").and_then(|p| p.as_str()).map(String::from))
                .unwrap_or_else(|| format!("sub-agent {}", sub.agent_id));

            let sys_id = format!("m{}", history.len());
            history.push(Message {
                id: sys_id.clone(),
                role: MessageRole::System,
                parts: vec![Part {
                    id: format!("{}.p0", sys_id),
                    kind: PartKind::Text,
                    content: format!(
                        "Sub-agent '{}' was interrupted mid-task (iteration {}). You may re-delegate if needed.",
                        prompt, sub.iteration,
                    ),
                    prune_state: PruneState::Intact,
                }],
            });

            resume_store.mark_agent_done(&sub.agent_id);
        }

        if !active_subs.is_empty() {
            app.blocks.push(DisplayBlock::SystemMessage(format!(
                "{} interrupted sub-agent(s) noted in context.",
                active_subs.len()
            )));
        }
    } else {
        // No root agent state in .db
        let sys_id = format!("m{}", history.len());
        history.push(Message {
            id: sys_id.clone(),
            role: MessageRole::System,
            parts: vec![Part {
                id: format!("{}.p0", sys_id),
                kind: PartKind::Text,
                content: "Session resumed. Your REPL environment was reset — re-import modules and recreate any state you need.".to_string(),
                prune_state: PruneState::Intact,
            }],
        });
    }
}

/// Transform `@path` tokens in user input to `[file: /abs/path]` or `[directory: /abs/path]`.
/// Rules: `@` must be at start of input or preceded by whitespace.
/// The token runs until whitespace or end of string.
/// Non-existent paths are left as-is.
fn transform_at_references(input: &str) -> String {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut result = String::with_capacity(input.len());
    let mut i = 0;
    let bytes = input.as_bytes();

    while i < bytes.len() {
        if bytes[i] == b'@' {
            // Check: must be at start or preceded by whitespace
            let at_start = i == 0 || bytes[i - 1].is_ascii_whitespace();
            if at_start {
                // Find the end of the token (next whitespace or end)
                let token_start = i + 1;
                let mut token_end = token_start;
                while token_end < bytes.len() && !bytes[token_end].is_ascii_whitespace() {
                    token_end += 1;
                }
                let token = &input[token_start..token_end];
                if !token.is_empty() {
                    let path = if token.starts_with('/') {
                        PathBuf::from(token)
                    } else {
                        cwd.join(token)
                    };
                    if path.is_file() {
                        result.push_str(&format!("[file: {}]", path.display()));
                        i = token_end;
                        continue;
                    } else if path.is_dir() {
                        // Strip trailing slash for display
                        let display = path.to_string_lossy();
                        let display = display.trim_end_matches('/');
                        result.push_str(&format!("[directory: {}]", display));
                        i = token_end;
                        continue;
                    }
                }
            }
        }
        result.push(input[i..].chars().next().unwrap());
        i += input[i..].chars().next().unwrap().len_utf8();
    }
    result
}

/// Send a desktop notification that the agent finished.
fn notify_done() {
    // Ensure the icon exists in ~/.lash/
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let icon_path = PathBuf::from(&home).join(".lash").join("icon.svg");
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
