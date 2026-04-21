//! `lash-export` — render a persisted lash session as HTML or JSON.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;

use lash_export::{ExportFormat, SessionSelector, export};

#[derive(Debug, Parser)]
#[command(
    name = "lash-export",
    about = "Render a persisted lash session as HTML or JSON"
)]
struct Cli {
    /// Session id (resolved against $LASH_HOME/sessions) or a direct path to a .db file.
    session: String,

    /// Output format.
    #[arg(long, default_value = "html")]
    format: String,

    /// Output file. If omitted, writes to stdout.
    #[arg(long, short = 'o')]
    out: Option<PathBuf>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("lash-export: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let format = ExportFormat::parse(&cli.format)?;

    let selector_path = PathBuf::from(&cli.session);
    let selector = if selector_path.is_file() {
        SessionSelector::Path(selector_path.as_path())
    } else {
        SessionSelector::Id(cli.session.as_str())
    };

    let sessions_dir = default_sessions_dir();
    let rendered = export(selector, &sessions_dir, format, cli.out.as_deref())
        .with_context(|| "rendering session")?;

    if cli.out.is_none() {
        print!("{rendered}");
    }
    Ok(())
}

fn default_sessions_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LASH_HOME") {
        return PathBuf::from(dir).join("sessions");
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".lash")
        .join("sessions")
}
