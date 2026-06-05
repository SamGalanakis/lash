//! `lash-export` — render a persisted lash session as HTML or JSON.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;

use lash_export::{ExportFormat, export};

#[derive(Debug, Parser)]
#[command(
    name = "lash-export",
    about = "Render a persisted lash session as HTML or JSON"
)]
struct Cli {
    /// Direct path to the session .db file.
    db: PathBuf,

    /// Full provider trace JSONL for the session.
    #[arg(long)]
    trace: PathBuf,

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

    let rendered = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .with_context(|| "starting async runtime")?
        .block_on(export(&cli.db, &cli.trace, format, cli.out.as_deref()))
        .with_context(|| "rendering session")?;

    if cli.out.is_none() {
        print!("{rendered}");
    }
    Ok(())
}
