use std::io::BufRead;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{ArgAction, Parser, ValueEnum};
use lash_debug_cli_harness::{
    ExecutionMode, HarnessConfig, LiveHarness, repo_root_from_manifest_dir,
};

#[derive(Debug, Parser)]
#[command(name = "lash-debug-cli-harness")]
#[command(about = "Drive the real lash interactive TUI through a PTY and save logs/screenshots.")]
struct Cli {
    #[arg(long, value_enum, default_value_t = ExecutionModeArg::Standard)]
    execution_mode: ExecutionModeArg,
    #[arg(long)]
    repo_root: Option<PathBuf>,
    #[arg(long)]
    lash_bin: Option<PathBuf>,
    #[arg(long)]
    lash_home: Option<PathBuf>,
    #[arg(long)]
    working_dir: Option<PathBuf>,
    #[arg(long)]
    output_dir: Option<PathBuf>,
    #[arg(long)]
    model: Option<String>,
    #[arg(long, default_value_t = 40)]
    rows: u16,
    #[arg(long, default_value_t = 120)]
    cols: u16,
    #[arg(long, default_value_t = 45)]
    timeout_secs: u64,
    #[arg(long, default_value_t = 250)]
    snapshot_interval_ms: u64,
    #[arg(long = "no-build", action = ArgAction::SetFalse, default_value_t = true)]
    build_lash: bool,
    #[arg(long, default_value = "warn")]
    lash_log: String,
    #[arg(long, hide = true)]
    control: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ExecutionModeArg {
    Standard,
    Rlm,
}

impl From<ExecutionModeArg> for ExecutionMode {
    fn from(value: ExecutionModeArg) -> Self {
        match value {
            ExecutionModeArg::Standard => Self::Standard,
            ExecutionModeArg::Rlm => Self::Rlm,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let repo_root = cli
        .repo_root
        .unwrap_or(repo_root_from_manifest_dir(env!("CARGO_MANIFEST_DIR"))?);

    let mut config = HarnessConfig::new(repo_root);
    config.execution_mode = cli.execution_mode.into();
    config.lash_bin = cli.lash_bin;
    config.lash_home = cli.lash_home;
    config.working_dir = cli.working_dir;
    config.model = cli.model;
    config.rows = cli.rows;
    config.cols = cli.cols;
    config.timeout = Duration::from_secs(cli.timeout_secs);
    config.snapshot_interval = Duration::from_millis(cli.snapshot_interval_ms);
    config.output_dir = cli.output_dir.unwrap_or(config.output_dir);
    config.build_lash = cli.build_lash;
    config.lash_log = cli.lash_log;

    run_control_loop(config)
}

fn run_control_loop(config: HarnessConfig) -> Result<()> {
    let mut harness = LiveHarness::start(config)?;
    println!("HARNESS ready");
    print_artifact_paths(&harness);
    print_help_summary();

    let stdin = std::io::stdin();
    let mut pending_submit_raw_len = None;
    for line in stdin.lock().lines() {
        let line = line.context("read harness command")?;
        let command = line.trim_end();
        if command.is_empty() {
            continue;
        }
        match handle_control_command(&mut harness, command, &mut pending_submit_raw_len) {
            Ok(ControlFlow::Continue) => {}
            Ok(ControlFlow::QuitCleanly) => {
                let run = harness.finish_cleanly()?;
                println!("HARNESS exited");
                println!("screen_txt: {}", run.artifacts.screen_txt.display());
                println!("screen_svg: {}", run.artifacts.screen_svg.display());
                println!("screen_png: {}", run.artifacts.screen_png.display());
                return Ok(());
            }
            Ok(ControlFlow::Kill) => {
                let run = harness.kill()?;
                println!("HARNESS killed");
                println!("screen_txt: {}", run.artifacts.screen_txt.display());
                println!("screen_svg: {}", run.artifacts.screen_svg.display());
                println!("screen_png: {}", run.artifacts.screen_png.display());
                return Ok(());
            }
            Err(error) => {
                println!("HARNESS error: {error:#}");
            }
        }
    }

    let run = harness.kill()?;
    println!("HARNESS stdin closed; killed child");
    println!("screen_txt: {}", run.artifacts.screen_txt.display());
    println!("screen_svg: {}", run.artifacts.screen_svg.display());
    println!("screen_png: {}", run.artifacts.screen_png.display());
    Ok(())
}

enum ControlFlow {
    Continue,
    QuitCleanly,
    Kill,
}

fn handle_control_command(
    harness: &mut LiveHarness,
    command: &str,
    pending_submit_raw_len: &mut Option<usize>,
) -> Result<ControlFlow> {
    let (verb, rest) = command
        .split_once(' ')
        .map(|(verb, rest)| (verb, rest))
        .unwrap_or((command, ""));
    match verb {
        "type" | "paste" => {
            harness.type_text(rest)?;
            println!("HARNESS ok");
        }
        "send" => {
            harness.type_text(rest)?;
            let raw_len_before_submit = harness.raw_output_len();
            harness.press_key("Enter")?;
            *pending_submit_raw_len = Some(raw_len_before_submit);
            println!("HARNESS ok");
        }
        "key" => {
            let raw_len_before_submit = harness.raw_output_len();
            harness.press_key(rest)?;
            if is_enter_key(rest) {
                *pending_submit_raw_len = Some(raw_len_before_submit);
            }
            println!("HARNESS ok");
        }
        "wait" => {
            harness.wait_for_text(rest, Duration::from_secs(45))?;
            println!("HARNESS ok");
        }
        "idle" => {
            if let Some(raw_len_before_submit) = pending_submit_raw_len.take() {
                harness.wait_submitted_turn_idle(raw_len_before_submit, Duration::from_secs(45))?;
            } else {
                harness.wait_idle(Duration::from_secs(45))?;
            }
            println!("HARNESS ok");
        }
        "screen" => {
            let screen = harness.screen_text()?;
            println!("HARNESS screen <<'EOF'\n{screen}\nEOF");
        }
        "screenshot" => {
            let name = if rest.trim().is_empty() {
                "manual"
            } else {
                rest.trim()
            };
            let snapshot = harness.screenshot(name)?;
            harness.write_current_artifacts()?;
            println!("HARNESS ok");
            println!("text: {}", snapshot.text.display());
            println!("svg: {}", snapshot.svg.display());
            println!("png: {}", snapshot.png.display());
        }
        "artifacts" => {
            print_artifact_paths(harness);
        }
        "log" => {
            if let Some(path) = harness.lash_log_path() {
                println!("HARNESS log: {}", path.display());
            } else {
                println!("HARNESS log: no --lash-home was set");
            }
        }
        "help" => print_help(),
        "quit" | "exit" => return Ok(ControlFlow::QuitCleanly),
        "kill" => return Ok(ControlFlow::Kill),
        other => bail!("unknown control command `{other}`"),
    }
    Ok(ControlFlow::Continue)
}

fn print_artifact_paths(harness: &LiveHarness) {
    let artifacts = harness.artifacts();
    println!("output_dir: {}", artifacts.output_dir.display());
    println!("screen_txt: {}", artifacts.screen_txt.display());
    println!("screen_svg: {}", artifacts.screen_svg.display());
    println!("screen_png: {}", artifacts.screen_png.display());
    println!("terminal_ansi: {}", artifacts.terminal_ansi.display());
    println!("ui_trace_json: {}", artifacts.ui_trace_json.display());
}

fn print_help_summary() {
    println!(
        "commands: send TEXT | type TEXT | key NAME | idle | wait TEXT | screen | screenshot NAME | artifacts | log | quit | kill | help"
    );
}

fn print_help() {
    println!("HARNESS commands:");
    println!("  send TEXT        type TEXT then press Enter");
    println!("  type TEXT        write bytes without Enter");
    println!("  paste TEXT       alias of type");
    println!(
        "  key NAME         press Enter, Tab, Esc, Backspace, Delete, arrows, Home/End, PageUp/PageDown, Ctrl-X, Alt-Up"
    );
    println!("  idle             wait until a submitted turn settles back to Idle");
    println!("  wait TEXT        wait until TEXT appears on the visible screen");
    println!("  screen           print current visible screen text");
    println!("  screenshot NAME  save screens/NAME.txt, screens/NAME.svg, and screens/NAME.png");
    println!("  artifacts        print artifact paths");
    println!("  log              print lash.log path when --lash-home is set");
    println!("  quit             send /exit, save artifacts, exit harness");
    println!("  kill             kill child, save artifacts, exit harness");
}

fn is_enter_key(key: &str) -> bool {
    matches!(key.trim().to_ascii_lowercase().as_str(), "enter" | "return")
}
