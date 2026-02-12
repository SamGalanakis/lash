use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use kaml::*;
use kaml::tools::{CompositeTools, EditFile, FetchUrl, Glob, Grep, Ls, ReadFile, Shell, WebSearch, WriteFile};

#[derive(Parser)]
struct Args {
    /// OpenRouter API key
    #[arg(long, env = "OPENROUTER_API_KEY")]
    api_key: String,

    /// Tavily API key for web search
    #[arg(long, env = "TAVILY_API_KEY")]
    tavily_api_key: Option<String>,

    /// Model name
    #[arg(long, default_value = "google/gemini-3-flash-preview")]
    model: String,

    /// Base URL for the LLM API
    #[arg(long, default_value = "https://openrouter.ai/api/v1")]
    base_url: String,

    /// Max iterations per user message
    #[arg(long, default_value = "10")]
    max_iterations: usize,
}

/// Session logger that writes JSONL to ~/.kaml/sessions/
struct SessionLogger {
    file: std::fs::File,
    path: PathBuf,
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

        let mut logger = Self { file, path };

        // Write session header
        logger.write(&serde_json::json!({
            "type": "session_start",
            "ts": now.to_rfc3339(),
            "model": model,
            "cwd": std::env::current_dir().ok().map(|p| p.to_string_lossy().to_string()),
        }))?;

        Ok(logger)
    }

    fn write(&mut self, value: &serde_json::Value) -> anyhow::Result<()> {
        serde_json::to_writer(&mut self.file, value)?;
        self.file.write_all(b"\n")?;
        self.file.flush()?;
        Ok(())
    }

    fn log_user_input(&mut self, input: &str) {
        let _ = self.write(&serde_json::json!({
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
        let _ = self.write(&value);
    }

    fn path(&self) -> &std::path::Path {
        &self.path
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut tools = CompositeTools::new()
        .add(Shell::new())
        .add(FetchUrl::new())
        .add(ReadFile::new())
        .add(WriteFile::new())
        .add(EditFile::new())
        .add(Glob::new())
        .add(Grep::new())
        .add(Ls::new());
    if let Some(ref key) = args.tavily_api_key {
        tools = tools.add(WebSearch::new(key));
    }
    let tools: Arc<dyn ToolProvider> = Arc::new(tools);
    let session = Session::new(tools, SessionConfig::default()).await?;

    let config = AgentConfig {
        model: args.model.clone(),
        api_key: args.api_key.clone(),
        base_url: args.base_url.clone(),
        max_iterations: args.max_iterations,
        ..Default::default()
    };

    let mut agent = Agent::new(session, config);
    let mut logger = SessionLogger::new(&args.model)?;

    println!("kaml Agent Demo (model: {})", args.model);
    eprintln!("\x1b[90mSession log: {}\x1b[0m", logger.path().display());
    println!("Type a message, or 'quit' to exit.\n");

    let mut history: Vec<agent::ChatMsg> = Vec::new();

    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "quit" {
            break;
        }

        logger.log_user_input(input);

        history.push(agent::ChatMsg {
            role: "user".to_string(),
            content: input.to_string(),
        });

        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<AgentEvent>(100);

        // Run agent in background, print events and log them
        let msgs = history.clone();
        let print_handle = tokio::spawn(async move {
            let mut events = Vec::new();
            while let Some(event) = event_rx.recv().await {
                match &event {
                    AgentEvent::TextDelta { content } => {
                        print!("{}", content);
                        std::io::stdout().flush().ok();
                    }
                    AgentEvent::CodeBlock { code } => {
                        println!("\x1b[90m┌─ python ───────────────────────────\x1b[0m");
                        for line in code.trim_end().lines() {
                            println!("\x1b[90m│\x1b[0m {}", line);
                        }
                    }
                    AgentEvent::CodeOutput { output, error } => {
                        if !output.is_empty() {
                            println!(
                                "\x1b[90m├─ stdout ──────────────────────────\x1b[0m"
                            );
                            for line in output.trim_end().lines() {
                                println!("\x1b[90m│ {}\x1b[0m", line);
                            }
                        }
                        if let Some(err) = error {
                            println!(
                                "\x1b[90m├─\x1b[0m \x1b[31merror\x1b[0m \x1b[90m─────────────────────────\x1b[0m"
                            );
                            for line in err.trim_end().lines() {
                                println!("\x1b[90m│\x1b[0m \x1b[31m{}\x1b[0m", line);
                            }
                        }
                        println!(
                            "\x1b[90m└───────────────────────────────────\x1b[0m"
                        );
                    }
                    AgentEvent::ToolCall {
                        name,
                        args: _,
                        result: _,
                        success,
                        duration_ms,
                    } => {
                        let icon = if *success { "+" } else { "x" };
                        println!(
                            "\x1b[90m  [{icon}] {name} ({}ms)\x1b[0m",
                            duration_ms
                        );
                    }
                    AgentEvent::LlmResponse {
                        iteration,
                        duration_ms,
                        ..
                    } => {
                        eprintln!(
                            "\x1b[90m  [llm] iteration {} ({}ms)\x1b[0m",
                            iteration, duration_ms
                        );
                    }
                    AgentEvent::Done => {
                        println!();
                    }
                    AgentEvent::Error { message } => {
                        eprintln!("\x1b[31mError: {}\x1b[0m", message);
                    }
                    _ => {}
                }
                events.push(event);
            }
            events
        });

        agent.run(msgs, event_tx).await;
        let events = print_handle.await?;

        // Log all events to session file, and collect assistant response for history
        let mut assistant_text = String::new();
        for event in &events {
            logger.log_event(event);
            if let AgentEvent::TextDelta { content } = event {
                assistant_text.push_str(content);
            }
        }

        // Add assistant response to history so the model sees the full conversation
        if !assistant_text.is_empty() {
            history.push(agent::ChatMsg {
                role: "assistant".to_string(),
                content: assistant_text,
            });
        }
    }

    Ok(())
}
