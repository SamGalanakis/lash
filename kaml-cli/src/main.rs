mod app;
mod event;
mod ui;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use clap::Parser;
use crossterm::event::{Event as TermEvent, KeyCode, KeyModifiers};
use kaml::agent::ChatMsg;
use kaml::tools::{
    CompositeTools, DelegateTask, EditFile, FetchUrl, Glob, Grep, Ls, ReadFile, Shell, WebSearch,
    WriteFile,
};
use kaml::*;
use ratatui::DefaultTerminal;
use tokio::sync::mpsc;

use app::{App, DisplayBlock};
use event::AppEvent;

#[derive(Parser)]
struct Args {
    /// OpenRouter API key
    #[arg(long, env = "OPENROUTER_API_KEY")]
    api_key: String,

    /// Tavily API key for web search
    #[arg(long, env = "TAVILY_API_KEY")]
    tavily_api_key: Option<String>,

    /// Model name
    #[arg(long, default_value = "z-ai/glm-5")]
    model: String,

    /// Base URL for the LLM API
    #[arg(long, default_value = "https://openrouter.ai/api/v1")]
    base_url: String,

    /// Max iterations per user message
    #[arg(long, default_value = "10")]
    max_iterations: usize,
}

struct SessionLogger {
    file: std::fs::File,
}

impl SessionLogger {
    fn new(model: &str) -> anyhow::Result<Self> {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let dir = PathBuf::from(home).join(".kaml").join("sessions");
        std::fs::create_dir_all(&dir)?;

        let now = chrono::Local::now();
        let filename = format!("{}.jsonl", now.format("%Y%m%d_%H%M%S"));
        let path = dir.join(&filename);
        let file = std::fs::File::create(&path)?;

        let mut logger = Self { file };
        logger.write_json(&serde_json::json!({
            "type": "session_start",
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
    let _ = crossterm::execute!(
        std::io::stdout(),
        crossterm::event::DisableMouseCapture,
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

    let args = Args::parse();

    // Build tools (same pattern as kaml-demo)
    let mut base = CompositeTools::new()
        .add(Shell::new())
        .add(FetchUrl::new())
        .add(ReadFile::new())
        .add(WriteFile::new())
        .add(EditFile::new())
        .add(Glob::new())
        .add(Grep::new())
        .add(Ls::new());
    if let Some(ref key) = args.tavily_api_key {
        base = base.add(WebSearch::new(key));
    }
    let base_tools: Arc<dyn ToolProvider> = Arc::new(base);

    let delegate = DelegateTask::new(
        Arc::clone(&base_tools),
        &args.model,
        &args.api_key,
        &args.base_url,
    );
    let tools: Arc<dyn ToolProvider> = Arc::new(
        CompositeTools::new()
            .add_arc(Arc::clone(&base_tools))
            .add(delegate),
    );
    let session = Session::new(tools, SessionConfig::default()).await?;

    let config = AgentConfig {
        model: args.model.clone(),
        api_key: args.api_key.clone(),
        base_url: args.base_url.clone(),
        max_iterations: args.max_iterations,
        ..Default::default()
    };

    let agent = Agent::new(session, config);
    let mut logger = SessionLogger::new(&args.model)?;

    // Install panic hook that restores the terminal
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        cleanup_terminal();
        default_hook(info);
    }));

    // Initialize terminal
    let terminal = ratatui::init();
    crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture)?;

    let result = run_app(terminal, agent, &mut logger, &args).await;

    cleanup_terminal();

    result
}

/// Returned by the spawned agent task so we can reclaim ownership.
struct AgentRunResult {
    agent: Agent,
    history: Vec<ChatMsg>,
}

async fn run_app(
    mut terminal: DefaultTerminal,
    agent: Agent,
    logger: &mut SessionLogger,
    args: &Args,
) -> anyhow::Result<()> {
    let mut app = App::new(args.model.clone());
    let mut history: Vec<ChatMsg> = Vec::new();
    let mut agent = Some(agent);

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
            if tick_tx
                .send(AppEvent::Terminal(TermEvent::FocusGained))
                .is_err()
            {
                break;
            }
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
                    agent_return_rx = None;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    app.running = false;
                    agent_return_rx = None;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {}
            }
        }

        // Draw
        terminal.draw(|frame| ui::draw(frame, &app))?;

        // Wait for next event
        let event = match app_rx.recv().await {
            Some(e) => e,
            None => break,
        };

        match event {
            AppEvent::Terminal(TermEvent::Key(key)) => {
                // CTRL+C: quit
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && key.code == KeyCode::Char('c')
                {
                    break;
                }

                // CTRL+O: toggle code expand (works in both modes)
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && key.code == KeyCode::Char('o')
                {
                    app.toggle_code_expand();
                    continue;
                }

                // Don't accept text input while agent is running
                if app.running {
                    continue;
                }

                match key.code {
                    KeyCode::Enter => {
                        let input = app.take_input();
                        if input.is_empty() {
                            continue;
                        }
                        if input == "quit" || input == "exit" {
                            break;
                        }

                        // /model command: switch model
                        if let Some(new_model) = input.strip_prefix("/model ") {
                            let new_model = new_model.trim().to_string();
                            if !new_model.is_empty() {
                                if let Some(ag) = agent.as_mut() {
                                    ag.set_model(new_model.clone());
                                }
                                app.model = new_model;
                            }
                            continue;
                        }

                        app.blocks.push(DisplayBlock::UserInput(input.clone()));
                        app.scroll_to_bottom();
                        app.running = true;
                        app.iteration = 0;

                        logger.log_user_input(&input);

                        history.push(ChatMsg {
                            role: "user".to_string(),
                            content: input,
                        });

                        // Create agent event channel that forwards to app events
                        let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(100);
                        let fwd_tx = app_tx.clone();
                        tokio::spawn(async move {
                            while let Some(ev) = event_rx.recv().await {
                                if fwd_tx.send(AppEvent::Agent(ev)).is_err() {
                                    break;
                                }
                            }
                        });

                        // Take agent out, spawn run task, get agent back via oneshot
                        let mut ag =
                            agent.take().expect("agent should be available when not running");
                        let msgs = history.clone();
                        let (return_tx, return_rx) = tokio::sync::oneshot::channel();
                        agent_return_rx = Some(return_rx);

                        tokio::spawn(async move {
                            let new_history = ag.run(msgs, event_tx).await;
                            let _ = return_tx.send(AgentRunResult {
                                agent: ag,
                                history: new_history,
                            });
                        });
                    }
                    KeyCode::Backspace => app.backspace(),
                    KeyCode::Delete => app.delete(),
                    KeyCode::Left => app.move_cursor_left(),
                    KeyCode::Right => app.move_cursor_right(),
                    KeyCode::Home => app.move_cursor_home(),
                    KeyCode::End => app.move_cursor_end(),
                    KeyCode::Up => app.history_up(),
                    KeyCode::Down => app.history_down(),
                    KeyCode::PageUp => app.scroll_up(10),
                    KeyCode::PageDown => {
                        let size = terminal.size()?;
                        let vh = size.height.saturating_sub(5) as usize;
                        let vw = size.width as usize;
                        app.scroll_down(10, vh, vw);
                    }
                    KeyCode::Char(c) => app.insert_char(c),
                    _ => {}
                }
            }
            AppEvent::Terminal(TermEvent::Mouse(mouse)) => {
                use crossterm::event::MouseEventKind;
                match mouse.kind {
                    MouseEventKind::ScrollUp => app.scroll_up(3),
                    MouseEventKind::ScrollDown => {
                        let size = terminal.size()?;
                        let vh = size.height.saturating_sub(5) as usize;
                        let vw = size.width as usize;
                        app.scroll_down(3, vh, vw);
                    }
                    _ => {}
                }
            }
            AppEvent::Terminal(_) => {
                if app.running {
                    app.tick += 1;
                }
            }
            AppEvent::Agent(event) => {
                logger.log_event(&event);
                app.handle_agent_event(event);
            }
        }
    }

    // Signal reader thread and tick timer to stop
    stop.store(true, Ordering::Relaxed);

    Ok(())
}
