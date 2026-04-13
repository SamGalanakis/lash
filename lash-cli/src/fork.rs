use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use lash::DynamicStateSnapshot;

use crate::app::UiResumeState;
use crate::session_log::{self, SessionLogger};

#[allow(clippy::too_many_arguments)]
async fn persist_parent_root_snapshot(
    runtime: &mut lash::LashRuntime,
    store: &lash::Store,
    ui_state: &UiResumeState,
    dynamic_state: &DynamicStateSnapshot,
) -> Result<()> {
    let mut state = runtime.export_state();
    state.execution_state_snapshot = runtime
        .snapshot_execution_state()
        .await
        .context("Failed to snapshot execution state for fork")?;
    crate::persist_root_session_state(store, &mut state, ui_state, dynamic_state);
    Ok(())
}

fn find_program_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

#[cfg(any(target_os = "macos", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DetectedTerminal {
    Ghostty,
    Kitty,
    WezTerm,
    Alacritty,
    AppleTerminal,
    ITerm2,
}

#[cfg(any(target_os = "macos", test))]
fn detect_terminal_from_hints(
    term_program: Option<&str>,
    has_kitty: bool,
    has_wezterm: bool,
    has_iterm: bool,
) -> Option<DetectedTerminal> {
    match term_program {
        Some("ghostty") => Some(DetectedTerminal::Ghostty),
        Some("kitty") => Some(DetectedTerminal::Kitty),
        Some("WezTerm") | Some("wezterm") => Some(DetectedTerminal::WezTerm),
        Some("alacritty") | Some("Alacritty") => Some(DetectedTerminal::Alacritty),
        Some("Apple_Terminal") => Some(DetectedTerminal::AppleTerminal),
        Some("iTerm.app") => Some(DetectedTerminal::ITerm2),
        _ if has_kitty => Some(DetectedTerminal::Kitty),
        _ if has_wezterm => Some(DetectedTerminal::WezTerm),
        _ if has_iterm => Some(DetectedTerminal::ITerm2),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
fn detect_current_terminal() -> Option<DetectedTerminal> {
    let term_program = std::env::var("TERM_PROGRAM").ok();
    detect_terminal_from_hints(
        term_program.as_deref(),
        std::env::var_os("KITTY_PID").is_some(),
        std::env::var_os("WEZTERM_PANE").is_some(),
        std::env::var_os("ITERM_SESSION_ID").is_some(),
    )
}

fn spawn_command(mut cmd: std::process::Command) -> bool {
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.spawn().is_ok()
}

#[cfg(target_os = "macos")]
fn run_command(mut cmd: std::process::Command) -> std::result::Result<(), String> {
    cmd.stdin(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    match cmd.output() {
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let detail = if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                format!("exit status {}", output.status)
            };
            Err(detail)
        }
        Err(err) => Err(err.to_string()),
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
fn spawn_with_launchers(exe: &Path, args: &[String], launchers: Vec<(&str, Vec<&str>)>) -> bool {
    for (launcher, prefix) in launchers {
        let Some(program) = find_program_on_path(launcher) else {
            continue;
        };
        let mut cmd = std::process::Command::new(program);
        cmd.args(prefix);
        cmd.arg(exe);
        cmd.args(args);
        if spawn_command(cmd) {
            return true;
        }
    }
    false
}

#[cfg(all(unix, not(target_os = "macos")))]
fn spawn_in_new_terminal_platform(exe: &Path, args: &[String]) -> Result<()> {
    let mut launchers: Vec<(&str, Vec<&str>)> = Vec::new();
    if let Ok(term_program) = std::env::var("TERM_PROGRAM") {
        match term_program.as_str() {
            "ghostty" => launchers.push(("ghostty", vec!["-e"])),
            "kitty" => launchers.push(("kitty", vec![])),
            "WezTerm" | "wezterm" => {
                launchers.push(("wezterm", vec!["start", "--always-new-process", "--"]))
            }
            _ => {}
        }
    }
    launchers.extend([
        ("ghostty", vec!["-e"]),
        ("kitty", vec![]),
        ("wezterm", vec!["start", "--always-new-process", "--"]),
        ("alacritty", vec!["-e"]),
        ("foot", vec![]),
        ("gnome-terminal", vec!["--"]),
        ("konsole", vec!["-e"]),
        ("xterm", vec!["-e"]),
    ]);
    if spawn_with_launchers(exe, args, launchers) {
        return Ok(());
    }
    Err(anyhow!(
        "Could not find a supported terminal launcher (tried ghostty, kitty, wezterm, alacritty, foot, gnome-terminal, konsole, xterm)"
    ))
}

#[cfg(target_os = "macos")]
fn shell_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".to_string();
    }
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

#[cfg(target_os = "macos")]
fn applescript_quote(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('\"', "\\\""))
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MacLauncher {
    GhosttyAppleScript,
    GhosttyCli,
    KittyCli,
    WezTermCli,
    AlacrittyCli,
    TerminalAppAppleScript,
    ITerm2AppleScript,
}

#[cfg(target_os = "macos")]
fn macos_command_line(exe: &Path, args: &[String]) -> String {
    let mut parts = Vec::with_capacity(args.len() + 1);
    parts.push(shell_quote(&exe.display().to_string()));
    parts.extend(args.iter().map(|arg| shell_quote(arg)));
    parts.join(" ")
}

#[cfg(target_os = "macos")]
fn spawn_with_single_launcher(
    exe: &Path,
    args: &[String],
    launcher: &str,
    prefix: &[&str],
) -> std::result::Result<(), String> {
    let Some(program) = find_program_on_path(launcher) else {
        return Err(format!("`{launcher}` is not available"));
    };
    let mut cmd = std::process::Command::new(program);
    cmd.args(prefix);
    cmd.arg(exe);
    cmd.args(args);
    if spawn_command(cmd) {
        Ok(())
    } else {
        Err(format!("failed to launch `{launcher}`"))
    }
}

#[cfg(target_os = "macos")]
fn spawn_macos_ghostty(exe: &Path, args: &[String]) -> std::result::Result<(), String> {
    let Some(program) = find_program_on_path("osascript") else {
        return Err("`osascript` is not available".to_string());
    };
    let command_line = macos_command_line(exe, args);

    let mut cmd = std::process::Command::new(program);
    cmd.arg("-e")
        .arg("tell application \"Ghostty\"")
        .arg("-e")
        .arg("activate")
        .arg("-e")
        .arg("set cfg to new surface configuration")
        .arg("-e")
        .arg(format!(
            "set command of cfg to {}",
            applescript_quote(&command_line)
        ))
        .arg("-e")
        .arg("set win to new window with configuration cfg")
        .arg("-e")
        .arg("activate window win")
        .arg("-e")
        .arg("end tell");
    run_command(cmd)
}

#[cfg(target_os = "macos")]
fn spawn_macos_iterm2(exe: &Path, args: &[String]) -> std::result::Result<(), String> {
    let Some(program) = find_program_on_path("osascript") else {
        return Err("`osascript` is not available".to_string());
    };
    let command_line = macos_command_line(exe, args);

    let mut cmd = std::process::Command::new(program);
    cmd.arg("-e")
        .arg("tell application \"iTerm2\"")
        .arg("-e")
        .arg("activate")
        .arg("-e")
        .arg(format!(
            "create window with default profile command {}",
            applescript_quote(&command_line)
        ))
        .arg("-e")
        .arg("end tell");
    run_command(cmd)
}

#[cfg(target_os = "macos")]
fn spawn_macos_terminal_app(exe: &Path, args: &[String]) -> std::result::Result<(), String> {
    let Some(program) = find_program_on_path("osascript") else {
        return Err("`osascript` is not available".to_string());
    };
    let command_line = macos_command_line(exe, args);

    let mut cmd = std::process::Command::new(program);
    cmd.arg("-e")
        .arg("tell application \"Terminal\"")
        .arg("-e")
        .arg("activate")
        .arg("-e")
        .arg(format!("do script {}", applescript_quote(&command_line)))
        .arg("-e")
        .arg("end tell");
    run_command(cmd)
}

#[cfg(target_os = "macos")]
fn push_unique_launcher(launchers: &mut Vec<MacLauncher>, launcher: MacLauncher) {
    if !launchers.contains(&launcher) {
        launchers.push(launcher);
    }
}

#[cfg(target_os = "macos")]
fn macos_launcher_label(launcher: MacLauncher) -> &'static str {
    match launcher {
        MacLauncher::GhosttyAppleScript => "Ghostty AppleScript",
        MacLauncher::GhosttyCli => "ghostty",
        MacLauncher::KittyCli => "kitty",
        MacLauncher::WezTermCli => "wezterm",
        MacLauncher::AlacrittyCli => "alacritty",
        MacLauncher::TerminalAppAppleScript => "Terminal.app AppleScript",
        MacLauncher::ITerm2AppleScript => "iTerm2 AppleScript",
    }
}

#[cfg(target_os = "macos")]
fn macos_detected_terminal_label(terminal: DetectedTerminal) -> &'static str {
    match terminal {
        DetectedTerminal::Ghostty => "Ghostty",
        DetectedTerminal::Kitty => "kitty",
        DetectedTerminal::WezTerm => "WezTerm",
        DetectedTerminal::Alacritty => "Alacritty",
        DetectedTerminal::AppleTerminal => "Terminal.app",
        DetectedTerminal::ITerm2 => "iTerm2",
    }
}

#[cfg(target_os = "macos")]
fn macos_launcher_order(current: Option<DetectedTerminal>) -> Vec<MacLauncher> {
    let mut launchers = Vec::new();
    if let Some(current) = current {
        match current {
            DetectedTerminal::Ghostty => {
                push_unique_launcher(&mut launchers, MacLauncher::GhosttyAppleScript);
                push_unique_launcher(&mut launchers, MacLauncher::GhosttyCli);
            }
            DetectedTerminal::Kitty => push_unique_launcher(&mut launchers, MacLauncher::KittyCli),
            DetectedTerminal::WezTerm => {
                push_unique_launcher(&mut launchers, MacLauncher::WezTermCli)
            }
            DetectedTerminal::Alacritty => {
                push_unique_launcher(&mut launchers, MacLauncher::AlacrittyCli)
            }
            DetectedTerminal::AppleTerminal => {
                push_unique_launcher(&mut launchers, MacLauncher::TerminalAppAppleScript)
            }
            DetectedTerminal::ITerm2 => {
                push_unique_launcher(&mut launchers, MacLauncher::ITerm2AppleScript)
            }
        }
    }

    for launcher in [
        MacLauncher::GhosttyAppleScript,
        MacLauncher::GhosttyCli,
        MacLauncher::KittyCli,
        MacLauncher::WezTermCli,
        MacLauncher::TerminalAppAppleScript,
    ] {
        push_unique_launcher(&mut launchers, launcher);
    }

    launchers
}

#[cfg(target_os = "macos")]
fn spawn_macos_launcher(
    launcher: MacLauncher,
    exe: &Path,
    args: &[String],
) -> std::result::Result<(), String> {
    match launcher {
        MacLauncher::GhosttyAppleScript => spawn_macos_ghostty(exe, args),
        MacLauncher::GhosttyCli => spawn_with_single_launcher(exe, args, "ghostty", &["-e"]),
        MacLauncher::KittyCli => spawn_with_single_launcher(exe, args, "kitty", &[]),
        MacLauncher::WezTermCli => spawn_with_single_launcher(
            exe,
            args,
            "wezterm",
            &["start", "--always-new-process", "--"],
        ),
        MacLauncher::AlacrittyCli => spawn_with_single_launcher(exe, args, "alacritty", &["-e"]),
        MacLauncher::TerminalAppAppleScript => spawn_macos_terminal_app(exe, args),
        MacLauncher::ITerm2AppleScript => spawn_macos_iterm2(exe, args),
    }
}

#[cfg(target_os = "macos")]
fn spawn_in_new_terminal_platform(exe: &Path, args: &[String]) -> Result<()> {
    let current_terminal = detect_current_terminal();
    let mut errors = Vec::new();

    for launcher in macos_launcher_order(current_terminal) {
        match spawn_macos_launcher(launcher, exe, args) {
            Ok(()) => return Ok(()),
            Err(err) => errors.push(format!(
                "{} failed: {}",
                macos_launcher_label(launcher),
                err
            )),
        }
    }

    let detected = current_terminal
        .map(|terminal| {
            format!(
                "Detected current terminal: {}. ",
                macos_detected_terminal_label(terminal)
            )
        })
        .unwrap_or_default();
    Err(anyhow!(
        concat!(
            "Could not launch a supported macOS terminal. ",
            "{}{} ",
            "If macOS Automation blocks terminal scripting, allow the launcher to control the terminal app in System Settings."
        ),
        detected,
        errors.join(" ")
    ))
}

#[cfg(test)]
mod tests {
    use super::{DetectedTerminal, detect_terminal_from_hints};

    #[test]
    fn detect_terminal_prefers_term_program() {
        assert_eq!(
            detect_terminal_from_hints(Some("ghostty"), true, true, true),
            Some(DetectedTerminal::Ghostty)
        );
        assert_eq!(
            detect_terminal_from_hints(Some("Apple_Terminal"), false, false, false),
            Some(DetectedTerminal::AppleTerminal)
        );
        assert_eq!(
            detect_terminal_from_hints(Some("iTerm.app"), false, false, false),
            Some(DetectedTerminal::ITerm2)
        );
    }

    #[test]
    fn detect_terminal_falls_back_to_terminal_specific_envs() {
        assert_eq!(
            detect_terminal_from_hints(None, true, false, false),
            Some(DetectedTerminal::Kitty)
        );
        assert_eq!(
            detect_terminal_from_hints(None, false, true, false),
            Some(DetectedTerminal::WezTerm)
        );
        assert_eq!(
            detect_terminal_from_hints(None, false, false, true),
            Some(DetectedTerminal::ITerm2)
        );
        assert_eq!(detect_terminal_from_hints(None, false, false, false), None);
    }
}

#[cfg(target_os = "windows")]
fn powershell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

#[cfg(target_os = "windows")]
fn spawn_windows_terminal(exe: &Path, args: &[String]) -> bool {
    let mut cmd = std::process::Command::new("wt.exe");
    cmd.arg("new-window");
    cmd.arg(exe);
    cmd.args(args);
    spawn_command(cmd)
}

#[cfg(target_os = "windows")]
fn spawn_windows_powershell(exe: &Path, args: &[String]) -> bool {
    let exe = powershell_quote(&exe.display().to_string());
    let script = if args.is_empty() {
        format!("Start-Process -FilePath {exe}")
    } else {
        let arg_list = args
            .iter()
            .map(|arg| powershell_quote(arg))
            .collect::<Vec<_>>()
            .join(", ");
        format!("Start-Process -FilePath {exe} -ArgumentList {arg_list}")
    };

    let mut cmd = std::process::Command::new("powershell.exe");
    cmd.arg("-NoProfile").arg("-Command").arg(script);
    spawn_command(cmd)
}

#[cfg(target_os = "windows")]
fn spawn_windows_cmd(exe: &Path, args: &[String]) -> bool {
    let mut cmd = std::process::Command::new("cmd.exe");
    cmd.arg("/C").arg("start").arg("").arg(exe);
    cmd.args(args);
    spawn_command(cmd)
}

#[cfg(target_os = "windows")]
fn spawn_in_new_terminal_platform(exe: &Path, args: &[String]) -> Result<()> {
    if spawn_windows_terminal(exe, args)
        || spawn_windows_powershell(exe, args)
        || spawn_windows_cmd(exe, args)
    {
        return Ok(());
    }
    Err(anyhow!(
        "Could not launch a supported Windows terminal (tried wt.exe, PowerShell Start-Process, cmd.exe start)"
    ))
}

#[cfg(not(any(unix, target_os = "windows")))]
fn spawn_in_new_terminal_platform(_exe: &Path, _args: &[String]) -> Result<()> {
    Err(anyhow!(
        "Opening a new terminal is not supported on this platform"
    ))
}

pub fn spawn_in_new_terminal(exe: &Path, args: &[String]) -> Result<()> {
    spawn_in_new_terminal_platform(exe, args)
}

fn forkable_live_graph(graph: &lash::SessionGraph) -> lash::SessionGraph {
    let mut prefix = Vec::new();
    let mut last_safe = lash::SessionGraph::default();
    let mut saw_safe_prefix = false;

    for node in graph.active_path_nodes() {
        prefix.push(node.clone());
        let candidate = lash::SessionGraph {
            nodes: prefix.clone(),
            leaf_node_id: prefix.last().map(|record| record.node_id.clone()),
        };
        if lash::messages_are_live_resume_safe(&candidate.project_messages()) {
            last_safe = candidate;
            saw_safe_prefix = true;
        }
    }

    if saw_safe_prefix {
        last_safe
    } else {
        lash::SessionGraph::default()
    }
}

#[allow(clippy::too_many_arguments)]
fn materialize_child_from_graph(
    child_store: &lash::Store,
    graph: &lash::SessionGraph,
    ui_state: &UiResumeState,
    persist_live_graph: bool,
) {
    let child_graph = graph.fork_current_path();
    child_store.save_session_graph(child_graph.clone());
    crate::ui_resume::save_ui_resume_state(child_store, ui_state);
    if persist_live_graph {
        child_store.save_live_session_graph(child_graph);
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn fork_current_session(
    runtime: Option<&mut lash::LashRuntime>,
    logger: &SessionLogger,
    ui_state: &UiResumeState,
    _provider: &lash::Provider,
    configured_model: &str,
    _context_window: u64,
    _model_variant: Option<&str>,
    _toolset_hash: &str,
    dynamic_state: &DynamicStateSnapshot,
) -> Result<(String, String)> {
    let live_graph_for_fork = if runtime.is_none() {
        logger
            .store()
            .load_live_session_graph()
            .map(|graph| forkable_live_graph(&graph))
    } else {
        None
    };
    if let Some(runtime) = runtime {
        persist_parent_root_snapshot(runtime, logger.store().as_ref(), ui_state, dynamic_state)
            .await?;
    }
    let parent_graph = logger.store().load_session_graph().unwrap_or_default();

    let sessions_dir = session_log::sessions_dir();
    let child_session_name = crate::generate_session_name(&sessions_dir);
    let child_filename = session_log::new_session_filename();
    let child_db_path = sessions_dir.join(&child_filename);
    let child_store = Arc::new(lash::Store::open(&child_db_path)?);
    let child_logger = SessionLogger::new(
        Arc::clone(&child_store),
        child_filename.clone(),
        configured_model,
        Some(uuid::Uuid::new_v4().to_string()),
        child_session_name.clone(),
    )?;
    child_logger.mark_as_child_of(&logger.session_id)?;
    if let Some(live_graph) = live_graph_for_fork.as_ref() {
        materialize_child_from_graph(child_store.as_ref(), live_graph, ui_state, true);
    } else {
        child_store.save_session_graph(parent_graph.fork_current_path());
    }

    Ok((child_filename, child_session_name))
}

#[cfg(test)]
mod fork_tests {
    use super::*;
    use crate::test_support::{EnvVarGuard, TempDirGuard, env_lock};
    use lash::provider::{Provider, ProviderOptions};
    use std::collections::{BTreeMap, BTreeSet};

    fn dummy_provider() -> Provider {
        Provider::OpenAiGeneric {
            api_key: "test".to_string(),
            base_url: "https://example.invalid/v1".to_string(),
            options: ProviderOptions::default(),
        }
    }

    fn empty_dynamic_state() -> DynamicStateSnapshot {
        DynamicStateSnapshot {
            base_generation: 0,
            tools: BTreeMap::new(),
            enabled_tools: BTreeSet::new(),
        }
    }

    fn persisted_graph(messages: Vec<lash::Message>, iteration: usize) -> lash::SessionGraph {
        let mut graph = lash::SessionGraph::from_projection(&messages, &[]);
        graph.record_runtime_state(
            &lash::PersistedSessionConfig {
                provider_id: dummy_provider().id().to_string(),
                configured_model: "gpt-test".to_string(),
                context_window: 1024,
                execution_mode: lash::ExecutionMode::Standard,
                context_approach: lash::ContextApproach::default(),
                model_variant: None,
            },
            &lash::PersistedTurnState {
                iteration,
                token_usage: lash::TokenUsage {
                    input_tokens: 10,
                    output_tokens: 3,
                    cached_input_tokens: 1,
                    reasoning_tokens: 2,
                },
                last_prompt_usage: None,
            },
            Some(&empty_dynamic_state()),
            None,
            None,
            &[],
        );
        graph
    }

    fn text_message(id: &str, role: lash::MessageRole, content: &str) -> lash::Message {
        lash::Message {
            id: id.to_string(),
            role,
            parts: vec![lash::Part {
                id: format!("{id}.p0"),
                kind: lash::PartKind::Text,
                content: content.to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: lash::PruneState::Intact,
            }],
            user_input: None,
            origin: None,
        }
    }

    fn unsafe_tool_call_message(id: &str, call_id: &str) -> lash::Message {
        lash::Message {
            id: id.to_string(),
            role: lash::MessageRole::Assistant,
            parts: vec![lash::Part {
                id: format!("{id}.p0"),
                kind: lash::PartKind::ToolCall,
                content: "{}".to_string(),
                attachment: None,
                tool_call_id: Some(call_id.to_string()),
                tool_name: Some("read_file".to_string()),
                prune_state: lash::PruneState::Intact,
            }],
            user_input: None,
            origin: None,
        }
    }

    #[tokio::test]
    async fn fork_clones_persisted_root_snapshot_without_runtime_when_no_live_snapshot() {
        let _env_guard = env_lock().lock().await;
        let temp = TempDirGuard::new("lash-fork-persisted-snapshot");
        let _lash_home = EnvVarGuard::set("LASH_HOME", temp.path());
        std::fs::create_dir_all(session_log::sessions_dir()).expect("sessions dir");

        let parent_filename = "parent.db".to_string();
        let parent_path = session_log::sessions_dir().join(&parent_filename);
        let parent_store = Arc::new(lash::Store::open(&parent_path).expect("parent store"));
        let parent_logger = SessionLogger::new(
            Arc::clone(&parent_store),
            parent_filename.clone(),
            "gpt-test",
            Some("parent-session".into()),
            "parent".into(),
        )
        .expect("parent logger");
        let messages = vec![lash::Message {
            id: "u1".to_string(),
            role: lash::MessageRole::User,
            parts: vec![lash::Part {
                id: "u1.p0".to_string(),
                kind: lash::PartKind::Text,
                content: "hello".to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: lash::PruneState::Intact,
            }],
            user_input: None,
            origin: None,
        }];
        parent_store.save_session_graph(persisted_graph(messages, 1));

        let (child_filename, _child_session_name) = fork_current_session(
            None,
            &parent_logger,
            &UiResumeState::default(),
            &dummy_provider(),
            "gpt-test",
            1024,
            None,
            "toolhash",
            &empty_dynamic_state(),
        )
        .await
        .expect("fork should succeed");

        let child_store = lash::Store::open(&session_log::sessions_dir().join(&child_filename))
            .expect("child store");
        let child_meta = child_store.load_session_meta().expect("child meta");
        assert_eq!(
            child_meta.parent_session_id.as_deref(),
            Some("parent-session")
        );

        let child_graph = child_store.load_session_graph().expect("child graph");
        let child_turn = child_graph.latest_turn_state().expect("child turn state");
        assert_eq!(child_turn.iteration, 1);
        assert!(child_store.load_live_session_graph().is_none());

        let child_messages = child_graph.project_messages();
        assert_eq!(child_messages.len(), 1);
        assert_eq!(child_messages[0].parts[0].content, "hello");
    }

    #[tokio::test]
    async fn fork_without_runtime_materializes_latest_live_snapshot() {
        let _env_guard = env_lock().lock().await;
        let temp = TempDirGuard::new("lash-fork-live-snapshot");
        let _lash_home = EnvVarGuard::set("LASH_HOME", temp.path());
        std::fs::create_dir_all(session_log::sessions_dir()).expect("sessions dir");

        let parent_filename = "parent.db".to_string();
        let parent_path = session_log::sessions_dir().join(&parent_filename);
        let parent_store = Arc::new(lash::Store::open(&parent_path).expect("parent store"));
        let parent_logger = SessionLogger::new(
            Arc::clone(&parent_store),
            parent_filename.clone(),
            "gpt-test",
            Some("parent-session".into()),
            "parent".into(),
        )
        .expect("parent logger");
        let base_messages = vec![lash::Message {
            id: "u1".to_string(),
            role: lash::MessageRole::User,
            parts: vec![lash::Part {
                id: "u1.p0".to_string(),
                kind: lash::PartKind::Text,
                content: "hello".to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: lash::PruneState::Intact,
            }],
            user_input: None,
            origin: None,
        }];
        parent_store.save_session_graph(persisted_graph(base_messages.clone(), 1));

        let live_messages = vec![
            base_messages[0].clone(),
            lash::Message {
                id: "a1".to_string(),
                role: lash::MessageRole::Assistant,
                parts: vec![lash::Part {
                    id: "a1.p0".to_string(),
                    kind: lash::PartKind::Text,
                    content: "latest coherent output".to_string(),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: lash::PruneState::Intact,
                }],
                user_input: None,
                origin: None,
            },
        ];
        crate::resume_snapshot::save_live_resume_snapshot(
            &parent_store,
            &lash::SessionStateEnvelope {
                session_id: crate::ROOT_SESSION_ID.to_string(),
                policy: lash::SessionPolicy {
                    execution_mode: lash::ExecutionMode::Standard,
                    ..lash::SessionPolicy::default()
                },
                session_graph: persisted_graph(live_messages.clone(), 2),
                messages: live_messages.clone(),
                tool_calls: Vec::new(),
                iteration: 2,
                token_usage: lash::TokenUsage {
                    input_tokens: 12,
                    output_tokens: 7,
                    cached_input_tokens: 1,
                    reasoning_tokens: 0,
                },
                last_prompt_usage: None,
                execution_state_snapshot: None,
                token_ledger: Vec::new(),
            },
            &UiResumeState::default(),
            &empty_dynamic_state(),
        )
        .expect("live snapshot");

        let (child_filename, _child_session_name) = fork_current_session(
            None,
            &parent_logger,
            &UiResumeState::default(),
            &dummy_provider(),
            "gpt-test",
            1024,
            None,
            "toolhash",
            &empty_dynamic_state(),
        )
        .await
        .expect("fork should succeed");

        let child_store = lash::Store::open(&session_log::sessions_dir().join(&child_filename))
            .expect("child store");
        let child_graph = child_store.load_session_graph().expect("child root graph");
        assert_eq!(
            child_graph
                .latest_turn_state()
                .expect("child turn state")
                .iteration,
            2
        );
        assert!(child_store.load_live_session_graph().is_some());

        let child_messages = child_graph.project_messages();
        assert_eq!(child_messages.len(), 2);
        assert_eq!(child_messages[1].parts[0].content, "latest coherent output");
    }

    #[tokio::test]
    async fn fork_without_runtime_uses_live_graph_even_when_resume_snapshot_is_unsafe() {
        let _env_guard = env_lock().lock().await;
        let temp = TempDirGuard::new("lash-fork-unsafe-live-graph");
        let _lash_home = EnvVarGuard::set("LASH_HOME", temp.path());
        std::fs::create_dir_all(session_log::sessions_dir()).expect("sessions dir");

        let parent_filename = "parent.db".to_string();
        let parent_path = session_log::sessions_dir().join(&parent_filename);
        let parent_store = Arc::new(lash::Store::open(&parent_path).expect("parent store"));
        let parent_logger = SessionLogger::new(
            Arc::clone(&parent_store),
            parent_filename.clone(),
            "gpt-test",
            Some("parent-session".into()),
            "parent".into(),
        )
        .expect("parent logger");

        let mut live_graph = persisted_graph(
            vec![text_message("u1", lash::MessageRole::User, "hello")],
            1,
        );
        live_graph.merge_active_projection(
            &[
                text_message("u1", lash::MessageRole::User, "hello"),
                unsafe_tool_call_message("a1", "call-1"),
            ],
            &[],
        );
        parent_store.save_live_session_graph(live_graph);

        let (child_filename, _child_session_name) = fork_current_session(
            None,
            &parent_logger,
            &UiResumeState::default(),
            &dummy_provider(),
            "gpt-test",
            1024,
            None,
            "toolhash",
            &empty_dynamic_state(),
        )
        .await
        .expect("fork should succeed");

        let child_store = lash::Store::open(&session_log::sessions_dir().join(&child_filename))
            .expect("child store");
        let child_graph = child_store.load_session_graph().expect("child root graph");
        let child_messages = child_graph.project_messages();
        assert_eq!(child_messages.len(), 1);
        assert_eq!(child_messages[0].parts[0].content, "hello");
        assert!(child_store.load_live_session_graph().is_some());
    }
}
