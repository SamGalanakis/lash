use std::io::Write;

use base64::Engine;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ClipboardEnv {
    pub ssh_connection: bool,
    pub ssh_tty: bool,
    pub tmux: bool,
    pub term: String,
    pub term_program: String,
}

pub(crate) fn copy_text_robustly(text: &str) -> Result<&'static str, String> {
    let mut errors = Vec::new();
    let mut osc52_ok = false;

    match copy_via_terminal_osc52(text) {
        Ok(()) => osc52_ok = true,
        Err(err) => errors.push(format!("osc52: {err}")),
    }

    match copy_via_system_clipboard(text) {
        Ok(()) => return Ok("system_clipboard"),
        Err(err) => errors.push(format!("system_clipboard: {err}")),
    }

    match copy_via_external_command(text) {
        Ok(method) => return Ok(method),
        Err(err) => errors.push(format!("external_command: {err}")),
    }

    if osc52_ok {
        return Ok("osc52");
    }

    Err(errors.join("; "))
}

fn copy_via_terminal_osc52(text: &str) -> Result<(), String> {
    if std::env::var_os("LASH_NO_OSC52").is_some() {
        return Err("disabled by LASH_NO_OSC52".to_string());
    }
    let env = current_clipboard_env();
    if !osc52_allowed_by_env(&env) {
        return Err("terminal environment does not look OSC52-capable".to_string());
    }

    let encoded = base64::engine::general_purpose::STANDARD.encode(text.as_bytes());
    let sequence = osc52_sequence_for(&encoded, &env);
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(sequence.as_bytes())
        .map_err(|err| format!("write failed: {err}"))?;
    stdout
        .flush()
        .map_err(|err| format!("flush failed: {err}"))?;
    Ok(())
}

pub(crate) fn osc52_allowed_by_env(env: &ClipboardEnv) -> bool {
    if env.ssh_connection || env.ssh_tty {
        return true;
    }

    let screen = env.term.starts_with("screen") || env.term.starts_with("tmux");

    env.tmux
        || screen
        || env.term.contains("xterm")
        || env.term.contains("rxvt")
        || env.term.contains("kitty")
        || env.term.contains("wezterm")
        || env.term.contains("alacritty")
        || env.term_program.contains("iterm")
        || env.term_program.contains("wezterm")
        || env.term_program.contains("apple_terminal")
        || env.term_program.contains("vscode")
}

pub(crate) fn osc52_sequence_for(encoded: &str, env: &ClipboardEnv) -> String {
    let base = format!("\x1b]52;c;{encoded}\x07");
    if env.tmux {
        format!("\x1bPtmux;\x1b{base}\x1b\\")
    } else if env.term.starts_with("screen") {
        format!("\x1bP{base}\x1b\\")
    } else {
        base
    }
}

fn current_clipboard_env() -> ClipboardEnv {
    ClipboardEnv {
        ssh_connection: std::env::var_os("SSH_CONNECTION").is_some(),
        ssh_tty: std::env::var_os("SSH_TTY").is_some(),
        tmux: std::env::var_os("TMUX").is_some(),
        term: std::env::var("TERM")
            .unwrap_or_default()
            .to_ascii_lowercase(),
        term_program: std::env::var("TERM_PROGRAM")
            .unwrap_or_default()
            .to_ascii_lowercase(),
    }
}

#[cfg(all(
    unix,
    not(any(target_os = "macos", target_os = "android", target_os = "emscripten"))
))]
fn copy_via_system_clipboard(text: &str) -> Result<(), String> {
    use std::sync::{Mutex, OnceLock};

    static CLIPBOARD: OnceLock<Mutex<Option<arboard::Clipboard>>> = OnceLock::new();
    let clipboard = CLIPBOARD.get_or_init(|| Mutex::new(None));
    let mut clipboard = clipboard
        .lock()
        .map_err(|_| "clipboard lock poisoned".to_string())?;
    if clipboard.is_none() {
        *clipboard = Some(arboard::Clipboard::new().map_err(|err| err.to_string())?);
    }

    clipboard
        .as_mut()
        .expect("clipboard initialized")
        .set_text(text.to_string())
        .map_err(|err| err.to_string())
}

#[cfg(any(
    target_os = "macos",
    target_os = "android",
    target_os = "emscripten",
    windows
))]
fn copy_via_system_clipboard(text: &str) -> Result<(), String> {
    let mut clipboard = arboard::Clipboard::new().map_err(|err| err.to_string())?;
    clipboard
        .set_text(text.to_string())
        .map_err(|err| err.to_string())
}

fn copy_via_external_command(text: &str) -> Result<&'static str, String> {
    #[cfg(target_os = "macos")]
    {
        return run_copy_command("pbcopy", &[], text).map(|()| "pbcopy");
    }

    #[cfg(target_os = "windows")]
    {
        return run_copy_command("clip.exe", &[], text).map(|()| "clip.exe");
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let candidates: [(&str, &[&str]); 4] = [
            ("wl-copy", &[]),
            ("xclip", &["-selection", "clipboard"]),
            ("xsel", &["--clipboard", "--input"]),
            ("pbcopy", &[]),
        ];
        let mut errors = Vec::new();
        for (cmd, args) in candidates {
            match run_copy_command(cmd, args, text) {
                Ok(()) => return Ok(cmd),
                Err(err) => errors.push(format!("{cmd}: {err}")),
            }
        }
        return Err(errors.join("; "));
    }

    #[allow(unreachable_code)]
    Err("no external clipboard command configured for this platform".to_string())
}

fn run_copy_command(cmd: &str, args: &[&str], text: &str) -> Result<(), String> {
    let mut child = std::process::Command::new(cmd)
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|err| err.to_string())?;

    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(text.as_bytes())
            .map_err(|err| format!("stdin write failed: {err}"))?;
    }

    let output = child.wait_with_output().map_err(|err| err.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            Err(format!("exited with status {}", output.status))
        } else {
            Err(stderr)
        }
    }
}
