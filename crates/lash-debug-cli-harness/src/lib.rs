use std::fmt;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use font8x8::UnicodeFonts;
use image::{Rgb, RgbImage};
use portable_pty::{CommandBuilder, PtySize};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum ExecutionMode {
    Standard,
    Rlm,
}

impl ExecutionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Rlm => "rlm",
        }
    }
}

impl fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug)]
pub struct HarnessConfig {
    pub repo_root: PathBuf,
    pub working_dir: Option<PathBuf>,
    pub lash_bin: Option<PathBuf>,
    pub lash_home: Option<PathBuf>,
    pub output_dir: PathBuf,
    pub execution_mode: ExecutionMode,
    pub model: Option<String>,
    pub rows: u16,
    pub cols: u16,
    pub timeout: Duration,
    pub snapshot_interval: Duration,
    pub build_lash: bool,
    pub lash_log: String,
}

impl HarnessConfig {
    pub fn new(repo_root: impl Into<PathBuf>) -> Self {
        Self {
            repo_root: repo_root.into(),
            working_dir: None,
            lash_bin: None,
            lash_home: None,
            output_dir: default_output_dir(),
            execution_mode: ExecutionMode::Standard,
            model: None,
            rows: 40,
            cols: 120,
            timeout: Duration::from_secs(45),
            snapshot_interval: Duration::from_millis(250),
            build_lash: true,
            lash_log: "warn".to_string(),
        }
    }

    pub fn resolve_lash_bin(&self) -> PathBuf {
        self.lash_bin
            .clone()
            .unwrap_or_else(|| self.repo_root.join("target/debug/lash"))
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct HarnessArtifacts {
    pub output_dir: PathBuf,
    pub terminal_ansi: PathBuf,
    pub screen_txt: PathBuf,
    pub screen_svg: PathBuf,
    pub screen_png: PathBuf,
    pub latest_screen_txt: PathBuf,
    pub latest_screen_svg: PathBuf,
    pub latest_screen_png: PathBuf,
    pub ui_trace_json: PathBuf,
    pub metadata_json: PathBuf,
}

impl HarnessArtifacts {
    fn new(output_dir: PathBuf) -> Self {
        let screens_dir = output_dir.join("screens");
        Self {
            terminal_ansi: output_dir.join("terminal.ansi"),
            screen_txt: output_dir.join("screen.txt"),
            screen_svg: output_dir.join("screen.svg"),
            screen_png: output_dir.join("screen.png"),
            latest_screen_txt: screens_dir.join("latest.txt"),
            latest_screen_svg: screens_dir.join("latest.svg"),
            latest_screen_png: screens_dir.join("latest.png"),
            ui_trace_json: output_dir.join("ui-trace.json"),
            metadata_json: output_dir.join("metadata.json"),
            output_dir,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct HarnessRun {
    pub artifacts: HarnessArtifacts,
    pub screen_text: String,
    pub raw_output_len: usize,
    pub elapsed_ms: u128,
    pub exit_status: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SnapshotPaths {
    pub text: PathBuf,
    pub svg: PathBuf,
    pub png: PathBuf,
}

#[derive(Debug, Serialize)]
struct HarnessMetadata<'a> {
    success: bool,
    execution_mode: ExecutionMode,
    model: Option<&'a str>,
    working_dir: Option<&'a Path>,
    lash_home: Option<&'a Path>,
    command: Vec<String>,
    elapsed_ms: u128,
    exit_status: Option<String>,
    artifacts: &'a HarnessArtifacts,
}

pub struct LiveHarness {
    config: HarnessConfig,
    artifacts: HarnessArtifacts,
    pty: PtyProcess,
    screen: ScreenCapture,
    command: Vec<String>,
    started: Instant,
    last_snapshot: Instant,
}

impl LiveHarness {
    pub fn start(config: HarnessConfig) -> Result<Self> {
        fs::create_dir_all(&config.output_dir)
            .with_context(|| format!("create output dir {}", config.output_dir.display()))?;
        if let Some(working_dir) = &config.working_dir {
            fs::create_dir_all(working_dir)
                .with_context(|| format!("create working dir {}", working_dir.display()))?;
        }
        if config.build_lash {
            build_lash(&config.repo_root)?;
        }

        let artifacts = HarnessArtifacts::new(config.output_dir.clone());
        fs::create_dir_all(artifacts.latest_screen_txt.parent().expect("screens dir"))?;

        let lash_bin = config.resolve_lash_bin();
        if !lash_bin.exists() {
            bail!(
                "lash binary does not exist at {}. Build it with `cargo build -p lash-cli` or pass --lash-bin.",
                lash_bin.display()
            );
        }

        let command = lash_command_args(&config, &artifacts);
        let pty = PtyProcess::spawn(&config, &lash_bin, &command)?;
        let last_snapshot = Instant::now()
            .checked_sub(config.snapshot_interval)
            .unwrap_or_else(Instant::now);
        let mut harness = Self {
            config,
            artifacts,
            pty,
            screen: ScreenCapture::new(),
            command,
            started: Instant::now(),
            last_snapshot,
        };
        harness.wait_for_ready()?;
        Ok(harness)
    }

    pub fn artifacts(&self) -> &HarnessArtifacts {
        &self.artifacts
    }

    pub fn lash_log_path(&self) -> Option<PathBuf> {
        self.config
            .lash_home
            .as_ref()
            .map(|home| home.join("lash.log"))
    }

    pub fn raw_output_len(&self) -> usize {
        self.pty.raw_output().len()
    }

    pub fn type_text(&mut self, text: &str) -> Result<()> {
        self.pty.write_bytes(text.as_bytes())
    }

    pub fn send_line(&mut self, line: &str) -> Result<()> {
        self.pty.write_line(line)
    }

    pub fn press_key(&mut self, key: &str) -> Result<()> {
        self.pty.write_bytes(&key_sequence(key)?)
    }

    pub fn wait_for_text(&mut self, needle: &str, timeout: Duration) -> Result<String> {
        self.wait_until(needle, timeout, |visible| visible.contains(needle))
    }

    pub fn wait_idle(&mut self, timeout: Duration) -> Result<String> {
        self.wait_until("Idle", timeout, |visible| visible.contains("Idle"))
    }

    pub fn wait_submitted_turn_idle(
        &mut self,
        raw_len_before_submit: usize,
        timeout: Duration,
    ) -> Result<String> {
        let deadline = Instant::now() + timeout;
        let mut saw_output_after_submit = false;
        loop {
            let visible = self.refresh_screen()?;
            saw_output_after_submit |= self.pty.raw_output().len() > raw_len_before_submit;
            if !visible.contains("Idle") {
                return self.wait_idle(timeout);
            }
            if saw_output_after_submit && input_prompt_is_empty(&visible) {
                return Ok(visible);
            }
            if let Some(status) = self.pty.try_wait()? {
                bail!("lash exited before submitted turn settled: {status:?}\n\n{visible}");
            }
            if Instant::now() >= deadline {
                self.write_current_artifacts()?;
                bail!("timed out waiting for submitted turn to settle\n\n{visible}");
            }
            thread::sleep(Duration::from_millis(50));
        }
    }

    pub fn screen_text(&mut self) -> Result<String> {
        self.refresh_screen()
    }

    pub fn screenshot(&mut self, name: &str) -> Result<SnapshotPaths> {
        let screen_text = self.refresh_screen()?;
        let safe_name = safe_artifact_name(name);
        let text = self
            .artifacts
            .output_dir
            .join("screens")
            .join(format!("{safe_name}.txt"));
        let svg = self
            .artifacts
            .output_dir
            .join("screens")
            .join(format!("{safe_name}.svg"));
        let png = self
            .artifacts
            .output_dir
            .join("screens")
            .join(format!("{safe_name}.png"));
        write_screen_artifacts(
            &text,
            &svg,
            &png,
            &screen_text,
            self.config.rows,
            self.config.cols,
        )?;
        Ok(SnapshotPaths { text, svg, png })
    }

    pub fn write_current_artifacts(&mut self) -> Result<String> {
        let raw = self.pty.raw_output();
        self.screen.process(&raw);
        let screen_text = self.screen.contents();
        write_terminal_artifacts(
            &self.artifacts,
            &raw,
            &screen_text,
            self.config.rows,
            self.config.cols,
        )?;
        Ok(screen_text)
    }

    pub fn finish_cleanly(mut self) -> Result<HarnessRun> {
        let screen_text = self.write_current_artifacts()?;
        self.send_line("/exit").context("send /exit to PTY")?;
        let status = self.pty.finish(Duration::from_secs(10), &mut self.screen)?;
        let exit_status = Some(format!("{status:?}"));
        let raw = self.pty.raw_output();
        write_terminal_artifacts(
            &self.artifacts,
            &raw,
            &screen_text,
            self.config.rows,
            self.config.cols,
        )?;
        write_metadata(
            &self.artifacts,
            &self.config,
            &self.command,
            self.started.elapsed(),
            status.success(),
            exit_status.clone(),
        )?;
        if !status.success() {
            bail!(
                "lash exited unsuccessfully: {status:?}. Artifacts: {}",
                self.artifacts.output_dir.display()
            );
        }
        Ok(HarnessRun {
            artifacts: self.artifacts.clone(),
            screen_text,
            raw_output_len: raw.len(),
            elapsed_ms: self.started.elapsed().as_millis(),
            exit_status,
        })
    }

    pub fn kill(mut self) -> Result<HarnessRun> {
        let screen_text = self.write_current_artifacts()?;
        self.pty.kill();
        let raw = self.pty.raw_output();
        write_metadata(
            &self.artifacts,
            &self.config,
            &self.command,
            self.started.elapsed(),
            false,
            None,
        )?;
        Ok(HarnessRun {
            artifacts: self.artifacts.clone(),
            screen_text,
            raw_output_len: raw.len(),
            elapsed_ms: self.started.elapsed().as_millis(),
            exit_status: None,
        })
    }

    fn wait_for_ready(&mut self) -> Result<String> {
        self.wait_until("input prompt", self.config.timeout, |visible| {
            visible.contains("Message") || visible.contains("Idle")
        })
    }

    fn wait_until(
        &mut self,
        label: &str,
        timeout: Duration,
        mut predicate: impl FnMut(&str) -> bool,
    ) -> Result<String> {
        let deadline = Instant::now() + timeout;
        loop {
            let visible = self.refresh_screen()?;
            if predicate(&visible) {
                return Ok(visible);
            }
            if let Some(status) = self.pty.try_wait()? {
                bail!("lash exited before `{label}` appeared: {status:?}\n\n{visible}");
            }
            if Instant::now() >= deadline {
                self.write_current_artifacts()?;
                bail!("timed out waiting for `{label}`\n\n{visible}");
            }
            thread::sleep(Duration::from_millis(50));
        }
    }

    fn refresh_screen(&mut self) -> Result<String> {
        let raw = self.pty.raw_output();
        self.screen.process(&raw);
        let visible = self.screen.contents();
        maybe_write_latest_snapshot(
            &self.artifacts,
            &visible,
            &self.config,
            &mut self.last_snapshot,
        )?;
        Ok(visible)
    }
}

fn build_lash(repo_root: &Path) -> Result<()> {
    let status = Command::new("cargo")
        .args(["build", "-p", "lash-cli"])
        .current_dir(repo_root)
        .status()
        .context("build lash CLI")?;
    if !status.success() {
        bail!("cargo build -p lash-cli failed with {status}");
    }
    Ok(())
}

fn lash_command_args(config: &HarnessConfig, artifacts: &HarnessArtifacts) -> Vec<String> {
    let mut args = Vec::new();
    if let Some(model) = &config.model {
        args.push("--model".to_string());
        args.push(model.clone());
    }
    args.extend([
        "--execution-mode".to_string(),
        config.execution_mode.as_str().to_string(),
        "--debug-ui-trace".to_string(),
        artifacts.ui_trace_json.display().to_string(),
        "--debug-ui-trace-interval-ms".to_string(),
        config.snapshot_interval.as_millis().to_string(),
    ]);
    args
}

struct PtyProcess {
    child: Box<dyn portable_pty::Child + Send>,
    writer: Box<dyn Write + Send>,
    output: Arc<Mutex<Vec<u8>>>,
    reader_thread: Option<thread::JoinHandle<()>>,
}

impl PtyProcess {
    fn spawn(config: &HarnessConfig, lash_bin: &Path, args: &[String]) -> Result<Self> {
        let pty_system = portable_pty::native_pty_system();
        let pair = pty_system
            .openpty(PtySize {
                rows: config.rows,
                cols: config.cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .context("open PTY")?;

        let mut cmd = CommandBuilder::new(lash_bin);
        cmd.args(args);
        let cwd = config.working_dir.as_deref().unwrap_or(&config.repo_root);
        cmd.cwd(cwd);
        if let Some(lash_home) = &config.lash_home {
            cmd.env("LASH_HOME", lash_home.as_os_str());
        }
        cmd.env("LASH_LOG", &config.lash_log);
        cmd.env("TERM", "xterm-256color");

        let child = pair.slave.spawn_command(cmd).context("spawn lash in PTY")?;
        let mut reader = pair.master.try_clone_reader().context("clone PTY reader")?;
        let writer = pair.master.take_writer().context("take PTY writer")?;
        drop(pair.slave);

        let output = Arc::new(Mutex::new(Vec::new()));
        let reader_output = Arc::clone(&output);
        let reader_thread = thread::spawn(move || {
            let mut buf = [0_u8; 8192];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        if let Ok(mut output) = reader_output.lock() {
                            output.extend_from_slice(&buf[..n]);
                        } else {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        Ok(Self {
            child,
            writer,
            output,
            reader_thread: Some(reader_thread),
        })
    }

    fn write_line(&mut self, line: &str) -> Result<()> {
        self.writer.write_all(line.as_bytes())?;
        self.writer.write_all(b"\r")?;
        self.writer.flush()?;
        Ok(())
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.writer.write_all(bytes)?;
        self.writer.flush()?;
        Ok(())
    }

    fn raw_output(&self) -> Vec<u8> {
        self.output.lock().expect("PTY output lock").clone()
    }

    fn try_wait(&mut self) -> Result<Option<portable_pty::ExitStatus>> {
        self.child.try_wait().context("poll lash child")
    }

    fn finish(
        &mut self,
        timeout: Duration,
        screen: &mut ScreenCapture,
    ) -> Result<portable_pty::ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.try_wait()? {
                screen.process(&self.raw_output());
                if let Some(reader_thread) = self.reader_thread.take() {
                    let _ = reader_thread.join();
                }
                return Ok(status);
            }
            if Instant::now() >= deadline {
                self.kill();
                bail!("timed out waiting for lash to exit");
            }
            screen.process(&self.raw_output());
            thread::sleep(Duration::from_millis(50));
        }
    }

    fn kill(&mut self) {
        let _ = self.child.kill();
        if let Some(reader_thread) = self.reader_thread.take() {
            let _ = reader_thread.join();
        }
    }
}

impl Drop for PtyProcess {
    fn drop(&mut self) {
        self.kill();
    }
}

struct ScreenCapture {
    parser: vt100::Parser,
    processed_len: usize,
}

impl ScreenCapture {
    fn new() -> Self {
        Self {
            parser: vt100::Parser::new(200, 240, 0),
            processed_len: 0,
        }
    }

    fn process(&mut self, raw: &[u8]) {
        if raw.len() > self.processed_len {
            self.parser.process(&raw[self.processed_len..]);
            self.processed_len = raw.len();
        }
    }

    fn contents(&self) -> String {
        trim_screen_rows(self.parser.screen().contents())
    }
}

fn maybe_write_latest_snapshot(
    artifacts: &HarnessArtifacts,
    screen_text: &str,
    config: &HarnessConfig,
    last_snapshot: &mut Instant,
) -> Result<()> {
    if last_snapshot.elapsed() < config.snapshot_interval {
        return Ok(());
    }
    *last_snapshot = Instant::now();
    write_screen_artifacts(
        &artifacts.latest_screen_txt,
        &artifacts.latest_screen_svg,
        &artifacts.latest_screen_png,
        screen_text,
        config.rows,
        config.cols,
    )
}

fn write_terminal_artifacts(
    artifacts: &HarnessArtifacts,
    raw: &[u8],
    screen_text: &str,
    rows: u16,
    cols: u16,
) -> Result<()> {
    fs::write(&artifacts.terminal_ansi, raw)?;
    write_screen_artifacts(
        &artifacts.screen_txt,
        &artifacts.screen_svg,
        &artifacts.screen_png,
        screen_text,
        rows,
        cols,
    )
}

fn write_screen_artifacts(
    text_path: &Path,
    svg_path: &Path,
    png_path: &Path,
    screen_text: &str,
    rows: u16,
    cols: u16,
) -> Result<()> {
    if let Some(parent) = text_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = svg_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = png_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(text_path, ensure_trailing_newline(screen_text))?;
    fs::write(svg_path, render_svg(screen_text, rows, cols))?;
    render_png(screen_text, rows, cols, png_path)?;
    Ok(())
}

fn write_metadata(
    artifacts: &HarnessArtifacts,
    config: &HarnessConfig,
    command: &[String],
    elapsed: Duration,
    success: bool,
    exit_status: Option<String>,
) -> Result<()> {
    let metadata = HarnessMetadata {
        success,
        execution_mode: config.execution_mode,
        model: config.model.as_deref(),
        working_dir: config.working_dir.as_deref(),
        lash_home: config.lash_home.as_deref(),
        command: command.to_vec(),
        elapsed_ms: elapsed.as_millis(),
        exit_status,
        artifacts,
    };
    fs::write(
        &artifacts.metadata_json,
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    Ok(())
}

fn render_svg(screen_text: &str, rows: u16, cols: u16) -> String {
    let cell_width = 9_u32;
    let line_height = 18_u32;
    let padding = 16_u32;
    let width = cols as u32 * cell_width + padding * 2;
    let height = rows as u32 * line_height + padding * 2;
    let mut svg = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#070707"/>
<style>
text {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size: 14px; fill: #e8e1d4; white-space: pre; }}
</style>
"##
    );
    for (index, line) in screen_text.lines().take(rows as usize).enumerate() {
        let y = padding + 14 + index as u32 * line_height;
        svg.push_str(&format!(
            r#"<text x="{padding}" y="{y}">{}</text>"#,
            escape_xml(line)
        ));
        svg.push('\n');
    }
    svg.push_str("</svg>\n");
    svg
}

fn render_png(screen_text: &str, rows: u16, cols: u16, path: &Path) -> Result<()> {
    let scale = 2_u32;
    let glyph_width = 8_u32 * scale;
    let line_height = 10_u32 * scale;
    let padding = 12_u32;
    let width = cols as u32 * glyph_width + padding * 2;
    let height = rows as u32 * line_height + padding * 2;
    let bg = Rgb([7, 7, 7]);
    let fg = Rgb([232, 225, 212]);
    let mut image = RgbImage::from_pixel(width, height, bg);

    for (line_index, line) in screen_text.lines().take(rows as usize).enumerate() {
        let y = padding + line_index as u32 * line_height;
        for (char_index, ch) in line.chars().take(cols as usize).enumerate() {
            let x = padding + char_index as u32 * glyph_width;
            draw_char(&mut image, x, y, screenshot_char(ch), scale, fg);
        }
    }

    image
        .save(path)
        .with_context(|| format!("write PNG screenshot {}", path.display()))
}

fn draw_char(image: &mut RgbImage, x: u32, y: u32, ch: char, scale: u32, color: Rgb<u8>) {
    let Some(glyph) = font8x8::BASIC_FONTS.get(ch) else {
        return;
    };
    for (row, bits) in glyph.iter().enumerate() {
        for col in 0..8 {
            if (bits >> col) & 1 == 1 {
                for dy in 0..scale {
                    for dx in 0..scale {
                        let px = x + col * scale + dx;
                        let py = y + row as u32 * scale + dy;
                        if px < image.width() && py < image.height() {
                            image.put_pixel(px, py, color);
                        }
                    }
                }
            }
        }
    }
}

fn screenshot_char(ch: char) -> char {
    match ch {
        '─' | '━' | '┄' | '┅' | '┈' | '┉' => '-',
        '│' | '┃' | '┊' | '┋' => '|',
        '┌' | '┐' | '└' | '┘' | '├' | '┤' | '┬' | '┴' | '┼' => '+',
        '●' | '○' => 'o',
        '■' | '□' => '#',
        '❯' | '›' => '>',
        '…' => '.',
        ch if ch.is_ascii() => ch,
        _ => '?',
    }
}

fn escape_xml(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn ensure_trailing_newline(text: &str) -> String {
    if text.ends_with('\n') {
        text.to_string()
    } else {
        format!("{text}\n")
    }
}

fn trim_screen_rows(screen: String) -> String {
    screen
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
}

fn input_prompt_is_empty(visible: &str) -> bool {
    visible
        .lines()
        .any(|line| line.trim_start().starts_with("❯ Message"))
}

pub fn key_sequence(name: &str) -> Result<Vec<u8>> {
    let lower = name.trim().to_ascii_lowercase();
    let bytes: &[u8] = match lower.as_str() {
        "enter" | "return" => b"\r",
        "tab" => b"\t",
        "esc" | "escape" => b"\x1b",
        "backspace" => b"\x7f",
        "delete" | "del" => b"\x1b[3~",
        "up" | "arrowup" => b"\x1b[A",
        "down" | "arrowdown" => b"\x1b[B",
        "right" | "arrowright" => b"\x1b[C",
        "left" | "arrowleft" => b"\x1b[D",
        "home" => b"\x1b[H",
        "end" => b"\x1b[F",
        "pageup" | "page-up" => b"\x1b[5~",
        "pagedown" | "page-down" => b"\x1b[6~",
        "alt-up" => b"\x1b[1;3A",
        "alt-down" => b"\x1b[1;3B",
        "alt-left" => b"\x1b[1;3D",
        "alt-right" => b"\x1b[1;3C",
        _ if lower.starts_with("ctrl-") && lower.len() == 6 => {
            let ch = lower.as_bytes()[5];
            if ch.is_ascii_lowercase() {
                return Ok(vec![ch - b'a' + 1]);
            }
            bail!("unknown key `{name}`");
        }
        _ => bail!("unknown key `{name}`"),
    };
    Ok(bytes.to_vec())
}

fn safe_artifact_name(name: &str) -> String {
    let mut safe = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            safe.push(ch);
        } else {
            safe.push('-');
        }
    }
    let safe = safe.trim_matches('-');
    if safe.is_empty() {
        "screen".to_string()
    } else {
        safe.to_string()
    }
}

fn default_output_dir() -> PathBuf {
    static NEXT_OUTPUT_DIR: AtomicUsize = AtomicUsize::new(1);
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sequence = NEXT_OUTPUT_DIR.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "lash-debug-cli-harness-{}-{stamp}-{sequence}",
        std::process::id()
    ))
}

pub fn repo_root_from_manifest_dir(manifest_dir: &str) -> Result<PathBuf> {
    let root = Path::new(manifest_dir)
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| anyhow::anyhow!("cannot derive repo root from {manifest_dir}"))?;
    Ok(root.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_names_map_to_terminal_bytes() {
        assert_eq!(key_sequence("Enter").unwrap(), b"\r");
        assert_eq!(key_sequence("Ctrl-C").unwrap(), b"\x03");
        assert_eq!(key_sequence("Alt-Up").unwrap(), b"\x1b[1;3A");
        assert!(key_sequence("Nope").is_err());
    }

    #[test]
    fn screenshot_names_are_filesystem_safe() {
        assert_eq!(safe_artifact_name("after first turn"), "after-first-turn");
        assert_eq!(safe_artifact_name("../bad"), "bad");
        assert_eq!(safe_artifact_name(""), "screen");
    }

    #[test]
    fn screen_rows_are_trimmed_without_dropping_structure() {
        assert_eq!(trim_screen_rows("a   \n\nb  ".to_string()), "a\n\nb");
    }

    #[test]
    fn empty_input_prompt_is_detected_without_matching_typed_text() {
        assert!(input_prompt_is_empty(
            "conversation\n/LASH  Idle\n ❯ Message · / for commands"
        ));
        assert!(!input_prompt_is_empty(
            "conversation\n/LASH  Idle\n ❯ QC scenario still in input"
        ));
    }

    #[test]
    fn svg_screenshot_escapes_terminal_text() {
        let svg = render_svg("a < b && c > d \"q\" 'z'", 4, 40);
        assert!(svg.contains("a &lt; b &amp;&amp; c &gt; d &quot;q&quot; &apos;z&apos;"));
        assert!(!svg.contains("a < b &&"));
    }

    #[test]
    fn png_screenshot_writes_png_file() {
        let temp = tempfile::tempdir().expect("temp dir");
        let png = temp.path().join("screen.png");
        render_png("hi", 2, 10, &png).expect("render png");
        let bytes = fs::read(png).expect("read png");
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    }
}
