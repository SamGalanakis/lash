use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use lash_core::{
    DynamicStateSnapshot, ExecutionMode, ExternalOpDef, ExternalOpKind, PluginError, PluginFactory,
    PluginRegistrar, PluginSessionContext, SessionConfigOverrides, SessionCreateRequest,
    SessionManager, SessionParam, SessionPlugin, SessionSnapshot, SessionStartPoint, ToolResult,
};
use serde_json::json;

use crate::session_log::{self, SessionLogger};

pub struct ForkPluginFactory;

struct ForkPlugin;

impl PluginFactory for ForkPluginFactory {
    fn id(&self) -> &'static str {
        "cli_fork"
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(ForkPlugin))
    }
}

impl SessionPlugin for ForkPlugin {
    fn id(&self) -> &'static str {
        "cli_fork"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.register_external_op(
            ExternalOpDef {
                name: "session.fork".to_string(),
                description: "Create a detached fork snapshot of the current session.".to_string(),
                kind: ExternalOpKind::Command,
                session_param: SessionParam::Optional,
                input_schema: json!({
                    "type": "object",
                    "additionalProperties": false
                }),
                output_schema: json!({
                    "type": "object",
                    "properties": {
                        "session_id": { "type": "string" },
                        "snapshot": { "type": "object" }
                    },
                    "required": ["session_id", "snapshot"],
                    "additionalProperties": false
                }),
            },
            Arc::new(|ctx, _args| {
                Box::pin(async move {
                    let start = if let Some(session_id) = ctx.session_id.clone() {
                        SessionStartPoint::ExistingSession { session_id }
                    } else {
                        SessionStartPoint::CurrentSession
                    };
                    match fork_snapshot_via_manager(ctx.host.as_ref(), start).await {
                        Ok((session_id, snapshot)) => ToolResult::ok(json!({
                            "session_id": session_id,
                            "snapshot": snapshot,
                        })),
                        Err(err) => ToolResult::err(json!(err.to_string())),
                    }
                })
            }),
        )
    }
}

async fn fork_snapshot_via_manager(
    manager: &dyn SessionManager,
    start: SessionStartPoint,
) -> Result<(String, SessionSnapshot), PluginError> {
    let handle = manager
        .create_session(SessionCreateRequest {
            agent_id: None,
            start,
            config_overrides: SessionConfigOverrides::default(),
            initial_messages: Vec::new(),
        })
        .await?;
    let snapshot = manager.snapshot_session(&handle.session_id).await?;
    manager.close_session(&handle.session_id).await?;
    Ok((handle.session_id, snapshot))
}

fn find_program_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
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
fn spawn_macos_terminal_app(exe: &Path, args: &[String]) -> bool {
    let Some(program) = find_program_on_path("osascript") else {
        return false;
    };
    let mut parts = Vec::with_capacity(args.len() + 1);
    parts.push(shell_quote(&exe.display().to_string()));
    parts.extend(args.iter().map(|arg| shell_quote(arg)));
    let command_line = parts.join(" ");

    let mut cmd = std::process::Command::new(program);
    cmd.arg("-e")
        .arg("tell application \"Terminal\"")
        .arg("-e")
        .arg("activate")
        .arg("-e")
        .arg(format!("do script {}", applescript_quote(&command_line)))
        .arg("-e")
        .arg("end tell");
    spawn_command(cmd)
}

#[cfg(target_os = "macos")]
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
    ]);
    if spawn_with_launchers(exe, args, launchers) || spawn_macos_terminal_app(exe, args) {
        return Ok(());
    }
    Err(anyhow!(
        "Could not launch a supported macOS terminal (tried ghostty, kitty, wezterm, Terminal.app via osascript)"
    ))
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

#[allow(clippy::too_many_arguments)]
pub async fn fork_current_session(
    runtime: &mut lash_core::LashRuntime,
    logger: &mut SessionLogger,
    provider: &lash_core::Provider,
    configured_model: &str,
    model_variant: Option<&str>,
    toolset_hash: &str,
    dynamic_state: &DynamicStateSnapshot,
) -> Result<(String, String)> {
    logger.flush()?;

    let fork_result = runtime
        .invoke_external("session.fork", json!({}), None)
        .await
        .map_err(|err| anyhow!(err.to_string()))?;
    if !fork_result.success {
        let message = fork_result
            .result
            .as_str()
            .unwrap_or("session.fork failed")
            .to_string();
        return Err(anyhow!(message));
    }

    let child_session_id = fork_result
        .result
        .get("session_id")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow!("session.fork did not return a session_id"))?
        .to_string();
    let snapshot_value = fork_result
        .result
        .get("snapshot")
        .cloned()
        .ok_or_else(|| anyhow!("session.fork did not return a snapshot"))?;
    let mut state: SessionSnapshot = serde_json::from_value(snapshot_value)
        .context("session.fork returned an invalid snapshot")?;

    if matches!(runtime.export_state().execution_mode, ExecutionMode::Repl) {
        let snapshot = runtime.snapshot_repl().await?;
        state.repl_snapshot = Some(snapshot);
    }

    let snapshot_hash = state.repl_snapshot.as_deref().map(crate::hash12);
    let sessions_dir = session_log::sessions_dir();
    let child_session_name = crate::generate_session_name(&sessions_dir);
    let mut child_logger = SessionLogger::new(
        configured_model,
        Some(child_session_id),
        child_session_name.clone(),
    )?;
    child_logger.clone_history_from(logger.filename())?;
    let child_filename = child_logger.filename().to_string();

    let child_db_path =
        sessions_dir.join(format!("{}.db", child_filename.trim_end_matches(".jsonl")));
    let child_store = lash_core::Store::open(&child_db_path)?;
    let execution_mode = state.execution_mode;
    let context_folding = state.context_folding;
    let prompt_hash = crate::latest_user_prompt_hash(&state.messages);
    crate::persist_root_agent_state(
        &child_store,
        &mut state,
        dynamic_state,
        provider,
        configured_model,
        execution_mode,
        context_folding,
        model_variant,
        toolset_hash,
        prompt_hash,
        snapshot_hash,
    );

    Ok((child_filename, child_session_name))
}
