use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use lash::oauth;
use lash::provider::{LashConfig, Provider};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph},
    Frame,
};

use crate::theme;

// ── State machine ───────────────────────────────────────────────────

enum SetupStep {
    SelectProvider { selected: usize },
    InputCredential { input: String, cursor: usize, error: Option<String>, verifier: Option<String> },
    CodexDeviceAuth {
        user_code: String,
        device_auth_id: String,
        interval: u64,
        error: Option<String>,
    },
    InputTavily { input: String, cursor: usize },
    Done,
}

struct SetupApp {
    step: SetupStep,
    tick: u64,
}

impl SetupApp {
    fn new() -> Self {
        Self {
            step: SetupStep::SelectProvider { selected: 0 },
            tick: 0,
        }
    }
}

// ── Public entry point ──────────────────────────────────────────────

pub async fn run_setup() -> anyhow::Result<LashConfig> {
    // Enter alternate screen + raw mode
    let mut terminal = ratatui::init();
    crossterm::execute!(
        std::io::stdout(),
        crossterm::style::Print("\x1b]11;rgb:0e/0d/0b\x1b\\")
    )?;

    let result = run_setup_inner(&mut terminal).await;

    // Restore terminal
    crossterm::execute!(
        std::io::stdout(),
        crossterm::style::Print("\x1b]111\x1b\\")
    )?;
    ratatui::restore();

    result
}

async fn run_setup_inner(terminal: &mut ratatui::DefaultTerminal) -> anyhow::Result<LashConfig> {
    let mut app = SetupApp::new();
    let mut provider: Option<Provider> = None;
    let mut tavily_key: Option<String> = None;

    loop {
        terminal.draw(|frame| draw_setup(frame, &app))?;

        // For CodexDeviceAuth: poll with timeout then poll the device auth endpoint
        if let SetupStep::CodexDeviceAuth {
            device_auth_id,
            user_code,
            interval,
            error,
        } = &mut app.step
        {
            let poll_secs = *interval;
            let poll_timeout = std::time::Duration::from_secs(poll_secs);

            if event::poll(poll_timeout)? {
                let ev = event::read()?;
                if let Event::Key(key) = ev {
                    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                        anyhow::bail!("Setup cancelled");
                    }
                    if key.code == KeyCode::Esc {
                        app.step = SetupStep::SelectProvider { selected: 1 };
                        continue;
                    }
                }
                // Tick for spinner
                app.tick += 1;
                continue;
            }

            // Timeout expired — poll the device auth endpoint
            app.tick += 1;
            let did = device_auth_id.clone();
            let uc = user_code.clone();
            match oauth::codex_poll_device_auth(&did, &uc).await {
                Ok(Some((auth_code, code_verifier))) => {
                    // Exchange for tokens
                    match oauth::codex_exchange_code(&auth_code, &code_verifier).await {
                        Ok(tokens) => {
                            provider = Some(Provider::Codex {
                                access_token: tokens.access_token,
                                refresh_token: tokens.refresh_token,
                                expires_at: tokens.expires_at,
                                account_id: tokens.account_id,
                            });
                            app.step = SetupStep::InputTavily {
                                input: String::new(),
                                cursor: 0,
                            };
                        }
                        Err(e) => {
                            *error = Some(format!("{}", e));
                        }
                    }
                }
                Ok(None) => {
                    // Still pending, continue polling
                }
                Err(e) => {
                    *error = Some(format!("{}", e));
                }
            }
            continue;
        }

        // Block until a key event (non-Codex steps)
        let ev = event::read()?;
        let Event::Key(key) = ev else { continue };

        // Ctrl+C always quits
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            anyhow::bail!("Setup cancelled");
        }

        match &mut app.step {
            SetupStep::SelectProvider { selected } => match key.code {
                KeyCode::Up | KeyCode::Char('k') => {
                    *selected = selected.saturating_sub(1);
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    *selected = (*selected + 1).min(2);
                }
                KeyCode::Enter => {
                    let sel = *selected;
                    match sel {
                        0 => {
                            // Claude — start OAuth
                            let (verifier, challenge) = oauth::generate_pkce();
                            let url = oauth::authorize_url(&challenge, &verifier);
                            let _ = open_browser(&url);
                            app.step = SetupStep::InputCredential {
                                input: String::new(),
                                cursor: 0,
                                error: None,
                                verifier: Some(verifier),
                            };
                        }
                        1 => {
                            // Codex — device code auth
                            match oauth::codex_request_device_code().await {
                                Ok(dc) => {
                                    let _ = open_browser(oauth::CODEX_DEVICE_VERIFY_URL);
                                    app.step = SetupStep::CodexDeviceAuth {
                                        user_code: dc.user_code,
                                        device_auth_id: dc.device_auth_id,
                                        interval: dc.interval,
                                        error: None,
                                    };
                                }
                                Err(e) => {
                                    app.step = SetupStep::CodexDeviceAuth {
                                        user_code: String::new(),
                                        device_auth_id: String::new(),
                                        interval: 5,
                                        error: Some(format!("{}", e)),
                                    };
                                }
                            }
                        }
                        _ => {
                            // OpenRouter
                            app.step = SetupStep::InputCredential {
                                input: String::new(),
                                cursor: 0,
                                error: None,
                                verifier: None,
                            };
                        }
                    }
                }
                _ => {}
            },

            SetupStep::InputCredential {
                input,
                cursor,
                error,
                verifier,
            } => match key.code {
                KeyCode::Esc => {
                    app.step = SetupStep::SelectProvider { selected: 0 };
                }
                KeyCode::Char(c) => {
                    input.insert(*cursor, c);
                    *cursor += c.len_utf8();
                    *error = None;
                }
                KeyCode::Backspace => {
                    if *cursor > 0 {
                        let prev = input[..*cursor]
                            .char_indices()
                            .last()
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        input.drain(prev..*cursor);
                        *cursor = prev;
                    }
                }
                KeyCode::Left => {
                    if *cursor > 0 {
                        *cursor = input[..*cursor]
                            .char_indices()
                            .last()
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                    }
                }
                KeyCode::Right => {
                    if *cursor < input.len() {
                        *cursor += input[*cursor..].chars().next().map(|c| c.len_utf8()).unwrap_or(0);
                    }
                }
                KeyCode::Enter => {
                    let val = input.trim().to_string();
                    if val.is_empty() {
                        *error = Some("Cannot be empty".into());
                    } else if verifier.is_some() {
                        // Claude OAuth exchange
                        let v = verifier.take().unwrap();
                        match oauth::exchange_code(&val, &v).await {
                            Ok(tokens) => {
                                provider = Some(Provider::Claude {
                                    access_token: tokens.access_token,
                                    refresh_token: tokens.refresh_token,
                                    expires_at: tokens.expires_at,
                                });
                                app.step = SetupStep::InputTavily {
                                    input: String::new(),
                                    cursor: 0,
                                };
                            }
                            Err(e) => {
                                *error = Some(format!("{}", e));
                                // Re-generate PKCE for retry
                                let (new_v, challenge) = oauth::generate_pkce();
                                let url = oauth::authorize_url(&challenge, &new_v);
                                let _ = open_browser(&url);
                                *verifier = Some(new_v);
                            }
                        }
                    } else {
                        // OpenRouter API key
                        provider = Some(Provider::OpenRouter {
                            api_key: val,
                            base_url: "https://openrouter.ai/api/v1".into(),
                        });
                        app.step = SetupStep::InputTavily {
                            input: String::new(),
                            cursor: 0,
                        };
                    }
                }
                _ => {}
            },

            SetupStep::CodexDeviceAuth { .. } => {
                // Handled by polling loop above; this branch is unreachable
                unreachable!();
            }

            SetupStep::InputTavily { input, cursor } => match key.code {
                KeyCode::Esc => {
                    // Skip tavily
                    tavily_key = None;
                    app.step = SetupStep::Done;
                }
                KeyCode::Char(c) => {
                    input.insert(*cursor, c);
                    *cursor += c.len_utf8();
                }
                KeyCode::Backspace => {
                    if *cursor > 0 {
                        let prev = input[..*cursor]
                            .char_indices()
                            .last()
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        input.drain(prev..*cursor);
                        *cursor = prev;
                    }
                }
                KeyCode::Left => {
                    if *cursor > 0 {
                        *cursor = input[..*cursor]
                            .char_indices()
                            .last()
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                    }
                }
                KeyCode::Right => {
                    if *cursor < input.len() {
                        *cursor += input[*cursor..].chars().next().map(|c| c.len_utf8()).unwrap_or(0);
                    }
                }
                KeyCode::Enter => {
                    let val = input.trim().to_string();
                    tavily_key = if val.is_empty() { None } else { Some(val) };
                    app.step = SetupStep::Done;
                }
                _ => {}
            },

            SetupStep::Done => unreachable!(),
        }

        if matches!(app.step, SetupStep::Done) {
            // Show "Authenticated!" briefly
            terminal.draw(|frame| draw_setup(frame, &app))?;
            std::thread::sleep(std::time::Duration::from_millis(500));
            break;
        }
    }

    let config = LashConfig {
        provider: provider.expect("provider must be set before Done"),
        tavily_api_key: tavily_key,
        delegate_models: None,
    };
    Ok(config)
}

// ── Rendering ───────────────────────────────────────────────────────

fn draw_setup(frame: &mut Frame, app: &SetupApp) {
    // Full background
    frame.render_widget(
        Block::default().style(Style::default().bg(theme::FORM)),
        frame.area(),
    );

    let logo_height = 8; // 5 logo + 1 trailing slash + 1 scribe + 1 tagline

    let step_height: u16 = match &app.step {
        SetupStep::SelectProvider { .. } => 9,  // blank + label + blank + 3 options + blank + help + pad
        SetupStep::InputCredential { error, .. } => {
            if error.is_some() { 10 } else { 8 }
        }
        SetupStep::CodexDeviceAuth { error, .. } => {
            if error.is_some() { 10 } else { 9 }
        }
        SetupStep::InputTavily { .. } => 7,
        SetupStep::Done => 4,
    };

    let chunks = Layout::vertical([
        Constraint::Min(0),                  // top spacer
        Constraint::Length(logo_height),      // logo
        Constraint::Length(step_height),      // step content
        Constraint::Min(0),                  // bottom spacer
    ])
    .split(frame.area());

    draw_logo(frame, chunks[1]);

    match &app.step {
        SetupStep::SelectProvider { selected } => draw_provider_select(frame, chunks[2], *selected),
        SetupStep::InputCredential {
            input,
            cursor,
            error,
            verifier,
        } => draw_credential_input(frame, chunks[2], input, *cursor, error.as_deref(), verifier.is_some()),
        SetupStep::CodexDeviceAuth {
            user_code, error, ..
        } => draw_codex_device_auth(frame, chunks[2], user_code, error.as_deref(), app.tick),
        SetupStep::InputTavily { input, cursor } => draw_tavily_input(frame, chunks[2], input, *cursor),
        SetupStep::Done => draw_done(frame, chunks[2]),
    }
}

fn draw_logo(frame: &mut Frame, area: Rect) {
    let chalk = Style::default().fg(theme::CHALK);
    let sodium = Style::default().fg(theme::SODIUM);

    let content_width = 30;
    let cx = (area.width as usize).saturating_sub(content_width) / 2;
    let pad = " ".repeat(cx);

    let logo: &[(&str, &str)] = &[
        ("██       ████   ██████  ", "  ██"),
        ("██      ██  ██  ██     ", "█  ██"),
        ("██      ██████  ██████", "██████"),
        ("██      ██  ██      █", " ██  ██"),
        ("██████  ██  ██  ████", "  ██  ██"),
    ];

    let mut lines: Vec<Line> = Vec::new();
    for &(before, after) in logo {
        lines.push(Line::from(vec![
            Span::styled(format!("{}{}", pad, before), chalk),
            Span::styled("██", sodium),
            Span::styled(after, chalk),
        ]));
    }
    // Trailing slash
    lines.push(Line::from(Span::styled(
        format!("{}                   ██", pad),
        sodium,
    )));
    // Scribe line
    lines.push(Line::from(vec![
        Span::styled(format!("{}──────────", pad), sodium),
        Span::styled("──────────", Style::default().fg(theme::ASH_MID)),
        Span::styled("──────────", Style::default().fg(theme::ASH)),
    ]));
    // Tagline
    lines.push(Line::from(Span::styled(
        format!("{}A G E N T  \u{b7}  R U N T I M E", pad),
        Style::default().fg(theme::ASH_TEXT),
    )));

    let paragraph = Paragraph::new(lines).style(Style::default().bg(theme::FORM));
    frame.render_widget(paragraph, area);
}

fn draw_provider_select(frame: &mut Frame, area: Rect, selected: usize) {
    let cx = center_pad(area.width as usize, 40);
    let pad = " ".repeat(cx);

    let options = [
        ("Claude", "Max/Pro subscription"),
        ("Codex", "ChatGPT Plus/Pro/Team"),
        ("OpenRouter", "API key"),
    ];

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!("{}Select provider:", pad),
        Style::default().fg(theme::ASH_TEXT),
    )));
    lines.push(Line::from(""));

    for (i, (name, desc)) in options.iter().enumerate() {
        if i == selected {
            lines.push(Line::from(vec![
                Span::styled(format!("{}\u{25b8} ", pad), Style::default().fg(theme::SODIUM)),
                Span::styled(
                    format!("{:<14}", name),
                    Style::default().fg(theme::CHALK).add_modifier(Modifier::BOLD),
                ),
                Span::styled(*desc, Style::default().fg(theme::CHALK_DIM)),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::styled(format!("{}  ", pad), Style::default()),
                Span::styled(
                    format!("{:<14}", name),
                    Style::default().fg(theme::ASH_TEXT),
                ),
                Span::styled(*desc, Style::default().fg(theme::ASH_MID)),
            ]));
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(""));
    // Help bar
    lines.push(Line::from(vec![
        Span::styled(format!("{}\u{2191}\u{2193}", pad), theme::help_key()),
        Span::styled(" navigate  ", theme::help_desc()),
        Span::styled("enter", theme::help_key()),
        Span::styled(" select", theme::help_desc()),
    ]));

    let paragraph = Paragraph::new(lines).style(Style::default().bg(theme::FORM));
    frame.render_widget(paragraph, area);
}

fn draw_credential_input(
    frame: &mut Frame,
    area: Rect,
    input: &str,
    cursor: usize,
    error: Option<&str>,
    is_claude: bool,
) {
    let box_width = 42usize;
    let cx = center_pad(area.width as usize, box_width);
    let pad = " ".repeat(cx);
    let inner_w = box_width.saturating_sub(4); // inside border + prompt char

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));

    if is_claude {
        lines.push(Line::from(Span::styled(
            format!("{}Authenticating with Claude...", pad),
            Style::default().fg(theme::ASH_TEXT),
        )));
        lines.push(Line::from(Span::styled(
            format!("{}Browser opened \u{2014} paste the code below.", pad),
            Style::default().fg(theme::ASH_TEXT),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            format!("{}Enter your OpenRouter API key.", pad),
            Style::default().fg(theme::ASH_TEXT),
        )));
        lines.push(Line::from(""));
    }

    // Label
    let label = if is_claude {
        "Authorization code"
    } else {
        "API key"
    };

    // Input box
    let top_border = format!(
        "{}\u{256d} {} {}\u{256e}",
        pad,
        label,
        "\u{2500}".repeat(box_width.saturating_sub(label.len() + 4)),
    );
    lines.push(Line::from(Span::styled(
        top_border,
        Style::default().fg(theme::ASH),
    )));

    // Input line: │ / text____│
    let display_text = visible_slice(input, inner_w, cursor);
    let text_pad = inner_w.saturating_sub(display_text.chars().count());
    lines.push(Line::from(vec![
        Span::styled(format!("{}\u{2502} ", pad), Style::default().fg(theme::ASH)),
        Span::styled(format!("{} ", theme::PROMPT_CHAR), Style::default().fg(theme::SODIUM)),
        Span::styled(display_text.to_string(), Style::default().fg(theme::CHALK_MID)),
        Span::styled(
            format!("{}\u{2502}", " ".repeat(text_pad)),
            Style::default().fg(theme::ASH),
        ),
    ]));

    let bottom_border = format!(
        "{}\u{2570}{}\u{256f}",
        pad,
        "\u{2500}".repeat(box_width.saturating_sub(2)),
    );
    lines.push(Line::from(Span::styled(
        bottom_border,
        Style::default().fg(theme::ASH),
    )));

    // Error message if any
    if let Some(err) = error {
        lines.push(Line::from(Span::styled(
            format!("{}  {}", pad, err),
            Style::default().fg(theme::ERROR),
        )));
    }

    lines.push(Line::from(""));
    // Help bar
    lines.push(Line::from(vec![
        Span::styled(format!("{}enter", pad), theme::help_key()),
        Span::styled(" submit  ", theme::help_desc()),
        Span::styled("esc", theme::help_key()),
        Span::styled(" back", theme::help_desc()),
    ]));

    let paragraph = Paragraph::new(lines).style(Style::default().bg(theme::FORM));
    frame.render_widget(paragraph, area);

    // Position cursor inside the input box
    let cursor_chars = input[..cursor].chars().count();
    let vis_start = visible_start(input, inner_w, cursor);
    let cursor_offset = cursor_chars.saturating_sub(vis_start);
    let cursor_x = (cx + 4 + cursor_offset) as u16; // pad + "│ / " = 4
    let cursor_y = area.y + if is_claude { 4 } else { 4 };
    if cursor_x < area.x + area.width && cursor_y < area.y + area.height {
        frame.set_cursor_position((area.x + cursor_x, cursor_y));
    }
}

fn draw_codex_device_auth(
    frame: &mut Frame,
    area: Rect,
    user_code: &str,
    error: Option<&str>,
    tick: u64,
) {
    let cx = center_pad(area.width as usize, 40);
    let pad = " ".repeat(cx);

    let spinner_chars = ["\u{2572}", "\u{2502}", "\u{2571}", "\u{2500}"];
    let spinner = spinner_chars[(tick as usize / 2) % spinner_chars.len()];

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!("{}Authenticating with Codex...", pad),
        Style::default().fg(theme::ASH_TEXT),
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(format!("{}Visit: ", pad), Style::default().fg(theme::ASH_TEXT)),
        Span::styled(
            "auth.openai.com/codex/device",
            Style::default().fg(theme::CHALK_MID),
        ),
    ]));

    if !user_code.is_empty() {
        lines.push(Line::from(vec![
            Span::styled(format!("{}Enter code: ", pad), Style::default().fg(theme::ASH_TEXT)),
            Span::styled(
                user_code,
                Style::default().fg(theme::SODIUM).add_modifier(Modifier::BOLD),
            ),
        ]));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(format!("{}{}", pad, spinner), Style::default().fg(theme::SODIUM)),
        Span::styled(
            "\u{2500} Waiting for authorization...",
            Style::default().fg(theme::ASH_MID),
        ),
    ]));

    if let Some(err) = error {
        lines.push(Line::from(Span::styled(
            format!("{}  {}", pad, err),
            Style::default().fg(theme::ERROR),
        )));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(format!("{}esc", pad), theme::help_key()),
        Span::styled(" back", theme::help_desc()),
    ]));

    let paragraph = Paragraph::new(lines).style(Style::default().bg(theme::FORM));
    frame.render_widget(paragraph, area);
}

fn draw_tavily_input(frame: &mut Frame, area: Rect, input: &str, cursor: usize) {
    let box_width = 42usize;
    let cx = center_pad(area.width as usize, box_width);
    let pad = " ".repeat(cx);
    let inner_w = box_width.saturating_sub(4);

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));

    // Label + input box
    let label = "Tavily API key (optional)";
    let top_border = format!(
        "{}\u{256d} {} {}\u{256e}",
        pad,
        label,
        "\u{2500}".repeat(box_width.saturating_sub(label.len() + 4)),
    );
    lines.push(Line::from(Span::styled(
        top_border,
        Style::default().fg(theme::ASH),
    )));

    let display_text = visible_slice(input, inner_w, cursor);
    let text_pad = inner_w.saturating_sub(display_text.chars().count());
    lines.push(Line::from(vec![
        Span::styled(format!("{}\u{2502} ", pad), Style::default().fg(theme::ASH)),
        Span::styled(format!("{} ", theme::PROMPT_CHAR), Style::default().fg(theme::SODIUM)),
        Span::styled(display_text.to_string(), Style::default().fg(theme::CHALK_MID)),
        Span::styled(
            format!("{}\u{2502}", " ".repeat(text_pad)),
            Style::default().fg(theme::ASH),
        ),
    ]));

    let bottom_border = format!(
        "{}\u{2570}{}\u{256f}",
        pad,
        "\u{2500}".repeat(box_width.saturating_sub(2)),
    );
    lines.push(Line::from(Span::styled(
        bottom_border,
        Style::default().fg(theme::ASH),
    )));

    lines.push(Line::from(""));
    // Help bar
    lines.push(Line::from(vec![
        Span::styled(format!("{}enter", pad), theme::help_key()),
        Span::styled(" submit  ", theme::help_desc()),
        Span::styled("esc", theme::help_key()),
        Span::styled(" skip", theme::help_desc()),
    ]));

    let paragraph = Paragraph::new(lines).style(Style::default().bg(theme::FORM));
    frame.render_widget(paragraph, area);

    // Position cursor
    let cursor_chars = input[..cursor].chars().count();
    let vis_start = visible_start(input, inner_w, cursor);
    let cursor_offset = cursor_chars.saturating_sub(vis_start);
    let cursor_x = (cx + 4 + cursor_offset) as u16;
    let cursor_y = area.y + 2; // input line is 3rd line (index 2)
    if cursor_x < area.x + area.width && cursor_y < area.y + area.height {
        frame.set_cursor_position((area.x + cursor_x, cursor_y));
    }
}

fn draw_done(frame: &mut Frame, area: Rect) {
    let cx = center_pad(area.width as usize, 20);
    let pad = " ".repeat(cx);

    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("{}Authenticated!", pad),
            Style::default().fg(theme::LICHEN).add_modifier(Modifier::BOLD),
        )),
    ];

    let paragraph = Paragraph::new(lines).style(Style::default().bg(theme::FORM));
    frame.render_widget(paragraph, area);
}

// ── Helpers ─────────────────────────────────────────────────────────

fn center_pad(viewport_width: usize, content_width: usize) -> usize {
    viewport_width.saturating_sub(content_width) / 2
}

/// Return the visible portion of text that fits in `width` chars around the cursor.
fn visible_slice(input: &str, width: usize, cursor_byte: usize) -> &str {
    if width == 0 {
        return "";
    }
    let chars: Vec<(usize, char)> = input.char_indices().collect();
    let total = chars.len();
    if total <= width {
        return input;
    }
    let cursor_char = input[..cursor_byte].chars().count();
    let start = if cursor_char > width.saturating_sub(1) {
        cursor_char - width + 1
    } else {
        0
    };
    let end = (start + width).min(total);
    let byte_start = chars[start].0;
    let byte_end = if end < total {
        chars[end].0
    } else {
        input.len()
    };
    &input[byte_start..byte_end]
}

/// Return the char index of the first visible character.
fn visible_start(input: &str, width: usize, cursor_byte: usize) -> usize {
    let total = input.chars().count();
    if total <= width {
        return 0;
    }
    let cursor_char = input[..cursor_byte].chars().count();
    if cursor_char > width.saturating_sub(1) {
        cursor_char - width + 1
    } else {
        0
    }
}

fn open_browser(url: &str) -> std::io::Result<()> {
    // Try common openers
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(url)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(url)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;
    }
    Ok(())
}
