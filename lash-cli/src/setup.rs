use std::collections::BTreeSet;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use lash::oauth;
use lash::provider::{LashConfig, Provider, ProviderKind};
use lash_tui::{Frame, Line, Modifier, Rect, Span, Style, Terminal};
use unicode_width::UnicodeWidthStr;

use crate::theme;

enum SetupStep {
    SelectProvider {
        selected: usize,
    },
    InputCredential {
        input: String,
        cursor: usize,
        error: Option<String>,
        verifier: Option<String>,
        auth_url: Option<String>,
        browser_error: Option<String>,
        mode: CredentialMode,
    },
    CodexDeviceAuth {
        user_code: String,
        device_auth_id: String,
        interval_secs: u64,
        error: Option<String>,
        copy_status: Option<String>,
    },
    InputBaseUrl {
        api_key: String,
        input: String,
        cursor: usize,
    },
    InputTavily {
        input: String,
        cursor: usize,
    },
    Done,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CredentialMode {
    GoogleOAuth,
    OpenAiGenericKey,
}

struct SetupApp {
    step: SetupStep,
    tick: u64,
    saved_kinds: BTreeSet<ProviderKind>,
    active_kind: Option<ProviderKind>,
}

impl SetupApp {
    fn new(existing: Option<&LashConfig>) -> Self {
        let active_kind = existing.map(LashConfig::active_provider_kind);
        let selected = active_kind
            .and_then(|kind| {
                ProviderKind::ALL
                    .iter()
                    .position(|candidate| *candidate == kind)
            })
            .unwrap_or(0);
        Self {
            step: SetupStep::SelectProvider { selected },
            tick: 0,
            saved_kinds: existing
                .map(|cfg| cfg.provider_kinds().into_iter().collect())
                .unwrap_or_default(),
            active_kind,
        }
    }
}

pub async fn run_setup_with_existing(existing: Option<&LashConfig>) -> anyhow::Result<LashConfig> {
    let mut terminal = Terminal::enter()?;
    let result = run_setup_inner(&mut terminal, existing).await;
    terminal.restore();
    result
}

async fn run_setup_inner(
    terminal: &mut Terminal,
    existing: Option<&LashConfig>,
) -> anyhow::Result<LashConfig> {
    let existing_config = existing.cloned();
    let mut app = SetupApp::new(existing_config.as_ref());
    let mut provider: Option<Provider> = None;
    let mut tavily_key = existing.and_then(|cfg| cfg.tavily_api_key().map(str::to_string));
    let existing_agent_models = existing.and_then(|cfg| cfg.agent_models.clone());

    loop {
        terminal.draw(|frame| draw_setup(frame, &app))?;

        if matches!(app.step, SetupStep::Done) {
            tokio::time::sleep(Duration::from_millis(350)).await;
            break;
        }

        if matches!(app.step, SetupStep::CodexDeviceAuth { .. }) {
            let interval_secs = match &app.step {
                SetupStep::CodexDeviceAuth { interval_secs, .. } => *interval_secs,
                _ => 1,
            };

            if event::poll(Duration::from_secs(interval_secs))? {
                let event = event::read()?;
                if let Event::Key(key) = event {
                    if key.kind == KeyEventKind::Release {
                        continue;
                    }
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('c')
                    {
                        anyhow::bail!("Setup cancelled");
                    }
                    if let SetupStep::CodexDeviceAuth {
                        user_code,
                        error,
                        copy_status,
                        ..
                    } = &mut app.step
                    {
                        match key.code {
                            KeyCode::Esc => {
                                app.step = SetupStep::SelectProvider { selected: 0 };
                            }
                            KeyCode::Char('c') if !user_code.is_empty() => {
                                *copy_status = None;
                                match copy_to_clipboard(user_code) {
                                    Ok(()) => {
                                        *copy_status = Some("Copied code to clipboard.".into())
                                    }
                                    Err(err) => *error = Some(err),
                                }
                                app.tick += 1;
                            }
                            _ => {}
                        }
                    }
                }
                app.tick += 1;
                continue;
            }

            app.tick += 1;
            let (device_auth_id, user_code) = match &app.step {
                SetupStep::CodexDeviceAuth {
                    device_auth_id,
                    user_code,
                    ..
                } => (device_auth_id.clone(), user_code.clone()),
                _ => unreachable!(),
            };

            match oauth::codex_poll_device_auth(&device_auth_id, &user_code).await {
                Ok(Some((auth_code, code_verifier))) => {
                    match oauth::codex_exchange_code(&auth_code, &code_verifier).await {
                        Ok(tokens) => {
                            provider = Some(Provider::Codex {
                                access_token: tokens.access_token,
                                refresh_token: tokens.refresh_token,
                                expires_at: tokens.expires_at,
                                account_id: tokens.account_id,
                                options: lash::provider::ProviderOptions::default(),
                            });
                            app.step = if tavily_key.is_some() {
                                SetupStep::Done
                            } else {
                                SetupStep::InputTavily {
                                    input: String::new(),
                                    cursor: 0,
                                }
                            };
                        }
                        Err(err) => {
                            if let SetupStep::CodexDeviceAuth { error, .. } = &mut app.step {
                                *error = Some(err.to_string());
                            }
                        }
                    }
                }
                Ok(None) => {}
                Err(err) => {
                    if let SetupStep::CodexDeviceAuth { error, .. } = &mut app.step {
                        *error = Some(err.to_string());
                    }
                }
            }
            continue;
        }

        let event = event::read()?;
        if let Event::Paste(text) = &event {
            match &mut app.step {
                SetupStep::InputCredential {
                    input,
                    cursor,
                    error,
                    ..
                } => {
                    insert_text(input, cursor, text);
                    *error = None;
                    continue;
                }
                SetupStep::InputBaseUrl { input, cursor, .. }
                | SetupStep::InputTavily { input, cursor } => {
                    insert_text(input, cursor, text);
                    continue;
                }
                _ => {}
            }
        }
        let Event::Key(key) = event else { continue };
        if key.kind == KeyEventKind::Release {
            continue;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            anyhow::bail!("Setup cancelled");
        }

        match &mut app.step {
            SetupStep::SelectProvider { selected } => match key.code {
                KeyCode::Up | KeyCode::Char('k') => {
                    *selected = selected.saturating_sub(1);
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    *selected = (*selected + 1).min(ProviderKind::ALL.len().saturating_sub(1));
                }
                KeyCode::Enter => {
                    let kind = ProviderKind::ALL[*selected];
                    if let Some(saved) = existing_config
                        .as_ref()
                        .and_then(|cfg| cfg.provider(kind))
                        .cloned()
                    {
                        provider = Some(saved);
                        app.active_kind = Some(kind);
                        app.step = if tavily_key.is_some() {
                            SetupStep::Done
                        } else {
                            SetupStep::InputTavily {
                                input: String::new(),
                                cursor: 0,
                            }
                        };
                    } else {
                        start_provider_flow(&mut app, kind, existing_config.as_ref()).await;
                    }
                }
                KeyCode::Char('r') => {
                    let kind = ProviderKind::ALL[*selected];
                    start_provider_flow(&mut app, kind, existing_config.as_ref()).await;
                }
                _ => {}
            },
            SetupStep::InputCredential {
                input,
                cursor,
                error,
                verifier,
                auth_url,
                browser_error,
                mode,
            } => match key.code {
                KeyCode::Esc => {
                    app.step = SetupStep::SelectProvider { selected: 0 };
                }
                KeyCode::Backspace => delete_left(input, cursor),
                KeyCode::Left => move_left(input, cursor),
                KeyCode::Right => move_right(input, cursor),
                KeyCode::Tab if *mode == CredentialMode::OpenAiGenericKey => {
                    app.step = SetupStep::InputBaseUrl {
                        api_key: String::new(),
                        input: String::new(),
                        cursor: 0,
                    };
                }
                KeyCode::Enter => {
                    let value = input.trim().to_string();
                    match mode {
                        CredentialMode::OpenAiGenericKey => {
                            app.step = SetupStep::InputBaseUrl {
                                api_key: value,
                                input: existing_openai_base_url(existing_config.as_ref()),
                                cursor: existing_openai_base_url(existing_config.as_ref()).len(),
                            };
                        }
                        CredentialMode::GoogleOAuth => {
                            if value.is_empty() {
                                *error = Some("Authorization code cannot be empty.".into());
                                continue;
                            }
                            let Some(verifier_value) = verifier.take() else {
                                *error = Some("OAuth state expired. Press Esc and retry.".into());
                                continue;
                            };
                            match oauth::google_exchange_code(&value, &verifier_value).await {
                                Ok(tokens) => {
                                    provider = Some(Provider::GoogleOAuth {
                                        access_token: tokens.access_token,
                                        refresh_token: tokens.refresh_token,
                                        expires_at: tokens.expires_at,
                                        project_id: std::env::var("GOOGLE_CLOUD_PROJECT")
                                            .ok()
                                            .or_else(|| {
                                                std::env::var("GOOGLE_CLOUD_PROJECT_ID").ok()
                                            }),
                                        options: lash::provider::ProviderOptions::default(),
                                    });
                                    app.step = if tavily_key.is_some() {
                                        SetupStep::Done
                                    } else {
                                        SetupStep::InputTavily {
                                            input: String::new(),
                                            cursor: 0,
                                        }
                                    };
                                }
                                Err(err) => {
                                    *error = Some(err.to_string());
                                    let (new_verifier, challenge) = oauth::generate_pkce();
                                    match oauth::google_authorize_url(&challenge) {
                                        Ok(url) => {
                                            persist_oauth_url(&url);
                                            *browser_error =
                                                open_browser(&url).err().map(|e| e.to_string());
                                            *auth_url = Some(url);
                                            *verifier = Some(new_verifier);
                                        }
                                        Err(auth_err) => {
                                            *auth_url = None;
                                            *browser_error = None;
                                            *verifier = Some(new_verifier);
                                            *error = Some(auth_err.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                KeyCode::Char(ch) => {
                    input.insert(*cursor, ch);
                    *cursor += ch.len_utf8();
                    *error = None;
                }
                _ => {}
            },
            SetupStep::InputBaseUrl {
                api_key,
                input,
                cursor,
            } => match key.code {
                KeyCode::Esc => {
                    app.step = SetupStep::SelectProvider { selected: 0 };
                }
                KeyCode::Backspace => delete_left(input, cursor),
                KeyCode::Left => move_left(input, cursor),
                KeyCode::Right => move_right(input, cursor),
                KeyCode::Enter => {
                    let base_url = input.trim().to_string();
                    if base_url.is_empty() {
                        continue;
                    }
                    provider = Some(Provider::OpenAiGeneric {
                        api_key: api_key.clone(),
                        base_url,
                        options: lash::provider::ProviderOptions::default(),
                    });
                    app.step = if tavily_key.is_some() {
                        SetupStep::Done
                    } else {
                        SetupStep::InputTavily {
                            input: String::new(),
                            cursor: 0,
                        }
                    };
                }
                KeyCode::Char(ch) => {
                    input.insert(*cursor, ch);
                    *cursor += ch.len_utf8();
                }
                _ => {}
            },
            SetupStep::InputTavily { input, cursor } => match key.code {
                KeyCode::Esc => {
                    tavily_key = None;
                    app.step = SetupStep::Done;
                }
                KeyCode::Backspace => delete_left(input, cursor),
                KeyCode::Left => move_left(input, cursor),
                KeyCode::Right => move_right(input, cursor),
                KeyCode::Enter => {
                    let value = input.trim().to_string();
                    tavily_key = if value.is_empty() { None } else { Some(value) };
                    app.step = SetupStep::Done;
                }
                KeyCode::Char(ch) => {
                    input.insert(*cursor, ch);
                    *cursor += ch.len_utf8();
                }
                _ => {}
            },
            SetupStep::CodexDeviceAuth { .. } | SetupStep::Done => {}
        }
    }

    let provider = provider.expect("provider must be selected before setup finishes");
    let mut config = existing_config.unwrap_or_else(|| LashConfig::new(provider.clone()));
    config.upsert_provider(provider.clone());
    config
        .set_active_provider_kind(provider.kind())
        .expect("active provider must exist after upsert");
    config.set_tavily_api_key(tavily_key);
    config.agent_models = existing_agent_models;
    Ok(config)
}

async fn start_provider_flow(
    app: &mut SetupApp,
    kind: ProviderKind,
    existing: Option<&LashConfig>,
) {
    match kind {
        ProviderKind::Codex => match oauth::codex_request_device_code().await {
            Ok(device) => {
                let _ = open_browser(oauth::CODEX_DEVICE_VERIFY_URL);
                app.step = SetupStep::CodexDeviceAuth {
                    user_code: device.user_code,
                    device_auth_id: device.device_auth_id,
                    interval_secs: device.interval,
                    error: None,
                    copy_status: None,
                };
            }
            Err(err) => {
                app.step = SetupStep::CodexDeviceAuth {
                    user_code: String::new(),
                    device_auth_id: String::new(),
                    interval_secs: 5,
                    error: Some(err.to_string()),
                    copy_status: None,
                };
            }
        },
        ProviderKind::GoogleOAuth => {
            let (verifier, challenge) = oauth::generate_pkce();
            match oauth::google_authorize_url(&challenge) {
                Ok(url) => {
                    persist_oauth_url(&url);
                    let browser_error = open_browser(&url).err().map(|err| err.to_string());
                    app.step = SetupStep::InputCredential {
                        input: String::new(),
                        cursor: 0,
                        error: None,
                        verifier: Some(verifier),
                        auth_url: Some(url),
                        browser_error,
                        mode: CredentialMode::GoogleOAuth,
                    };
                }
                Err(err) => {
                    app.step = SetupStep::InputCredential {
                        input: String::new(),
                        cursor: 0,
                        error: Some(err.to_string()),
                        verifier: Some(verifier),
                        auth_url: None,
                        browser_error: None,
                        mode: CredentialMode::GoogleOAuth,
                    };
                }
            }
        }
        ProviderKind::OpenAiGeneric => {
            let existing_key = existing_openai_key(existing);
            app.step = SetupStep::InputCredential {
                input: existing_key.clone(),
                cursor: existing_key.len(),
                error: None,
                verifier: None,
                auth_url: None,
                browser_error: None,
                mode: CredentialMode::OpenAiGenericKey,
            };
        }
    }
}

fn draw_setup(frame: &mut Frame<'_>, app: &SetupApp) {
    let area = frame.area();
    frame.clear(Style::default().bg(theme::FORM));
    if area.width < 24 || area.height < 10 {
        frame.write_text(
            2,
            2,
            "Enlarge the terminal to continue setup.",
            Style::default().fg(theme::CHALK_DIM),
            area.width.saturating_sub(4),
        );
        return;
    }

    draw_header(frame, area);
    let panel = centered_rect(
        area,
        area.width.saturating_sub(12).min(86),
        area.height.saturating_sub(8).min(24),
    );
    frame.draw_box(
        panel,
        Style::default().fg(theme::ASH_LIGHT),
        Some(Style::default().bg(theme::FORM_DEEP)),
    );

    match &app.step {
        SetupStep::SelectProvider { selected } => {
            draw_panel_title(frame, panel, "Provider Setup");
            let mut lines = vec![
                Line::from(vec![
                    Span::styled("Select a provider. ", Style::default().fg(theme::CHALK)),
                    Span::styled("Enter", theme::help_key()),
                    Span::styled(
                        " uses a saved login when one exists; ",
                        Style::default().fg(theme::CHALK_DIM),
                    ),
                    Span::styled("r", theme::help_key()),
                    Span::styled(" forces re-auth.", Style::default().fg(theme::CHALK_DIM)),
                ]),
                Line::from(""),
            ];

            for (idx, kind) in ProviderKind::ALL.iter().enumerate() {
                let selected_marker = if *selected == idx { "▸" } else { " " };
                let name_style = if *selected == idx {
                    Style::default()
                        .fg(theme::CHALK)
                        .add_modifier(Modifier::Bold)
                } else {
                    Style::default().fg(theme::CHALK_DIM)
                };
                let mut meta = kind.setup_description().to_string();
                if app.active_kind == Some(*kind) {
                    meta.push_str(" · active");
                } else if app.saved_kinds.contains(kind) {
                    meta.push_str(" · saved");
                }
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("{selected_marker} "),
                        Style::default()
                            .fg(theme::SODIUM)
                            .add_modifier(Modifier::Bold),
                    ),
                    Span::styled(format!("{:<22}", kind.setup_name()), name_style),
                    Span::styled(meta, Style::default().fg(theme::ASH_TEXT)),
                ]));
            }

            lines.push(Line::from(""));
            lines.push(help_line(&[
                ("↑↓", "move"),
                ("enter", "select"),
                ("r", "re-auth"),
                ("ctrl+c", "quit"),
            ]));
            draw_lines(frame, inner_panel(panel), &lines);
        }
        SetupStep::InputCredential {
            input,
            cursor,
            error,
            auth_url,
            browser_error,
            mode,
            ..
        } => {
            let (title, body, label, help) = match mode {
                CredentialMode::GoogleOAuth => (
                    "Google OAuth",
                    google_auth_body(auth_url.as_deref(), browser_error.as_deref()),
                    "Paste code or callback URL",
                    vec![("enter", "submit"), ("esc", "back"), ("ctrl+c", "quit")],
                ),
                CredentialMode::OpenAiGenericKey => (
                    "OpenAI-Compatible",
                    vec!["Paste an API key, or press Tab to skip if your endpoint does not require one.".to_string()],
                    "API key",
                    vec![("enter", "next"), ("tab", "skip"), ("esc", "back"), ("ctrl+c", "quit")],
                ),
            };
            draw_input_panel(
                frame,
                panel,
                InputPanelView {
                    title,
                    body: &body,
                    label,
                    input,
                    cursor: *cursor,
                    error: error.as_deref(),
                    help: &help,
                },
            );
        }
        SetupStep::CodexDeviceAuth {
            user_code,
            error,
            copy_status,
            ..
        } => {
            draw_panel_title(frame, panel, "Codex OAuth");
            let spinner = ["╱", "│", "╲", "─"][(app.tick as usize / 2) % 4];
            let mut lines = vec![
                Line::from("Open the device login page and approve this machine."),
                Line::from(""),
                Line::from(vec![
                    Span::styled("URL: ", Style::default().fg(theme::ASH_TEXT)),
                    Span::styled(
                        "auth.openai.com/codex/device",
                        Style::default().fg(theme::CHALK),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Code: ", Style::default().fg(theme::ASH_TEXT)),
                    Span::styled(
                        user_code.to_string(),
                        Style::default()
                            .fg(theme::SODIUM)
                            .add_modifier(Modifier::Bold),
                    ),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled(spinner.to_string(), Style::default().fg(theme::SODIUM)),
                    Span::styled(
                        " Waiting for authorization…",
                        Style::default().fg(theme::CHALK_DIM),
                    ),
                ]),
            ];
            if let Some(status) = copy_status {
                lines.push(Line::from(Span::styled(
                    status.to_string(),
                    Style::default().fg(theme::LICHEN),
                )));
            }
            if let Some(error) = error {
                lines.push(Line::from(Span::styled(
                    error.to_string(),
                    Style::default().fg(theme::ERROR),
                )));
            }
            lines.push(Line::from(""));
            lines.push(help_line(&[
                ("c", "copy code"),
                ("esc", "back"),
                ("ctrl+c", "quit"),
            ]));
            draw_lines(frame, inner_panel(panel), &lines);
        }
        SetupStep::InputBaseUrl { input, cursor, .. } => {
            draw_input_panel(
                frame,
                panel,
                InputPanelView {
                    title: "OpenAI-Compatible",
                    body: &[
                        "Enter the API base URL for your endpoint.".to_string(),
                        "Example: https://api.openai.com/v1".to_string(),
                    ],
                    label: "Base URL",
                    input,
                    cursor: *cursor,
                    error: None,
                    help: &[("enter", "save"), ("esc", "back"), ("ctrl+c", "quit")],
                },
            );
        }
        SetupStep::InputTavily { input, cursor } => {
            draw_input_panel(
                frame,
                panel,
                InputPanelView {
                    title: "Optional Web Search",
                    body: &["Add a Tavily API key now, or press Esc to skip.".to_string()],
                    label: "Tavily API key (optional)",
                    input,
                    cursor: *cursor,
                    error: None,
                    help: &[("enter", "save"), ("esc", "skip"), ("ctrl+c", "quit")],
                },
            );
        }
        SetupStep::Done => {
            draw_panel_title(frame, panel, "Ready");
            let lines = vec![
                Line::from(""),
                Line::from(Span::styled(
                    "Authenticated.",
                    Style::default()
                        .fg(theme::LICHEN)
                        .add_modifier(Modifier::Bold),
                )),
            ];
            draw_lines(frame, inner_panel(panel), &lines);
        }
    }
}

fn draw_header(frame: &mut Frame<'_>, area: Rect) {
    frame.write_text(
        3,
        1,
        "lash",
        Style::default()
            .fg(theme::SODIUM)
            .add_modifier(Modifier::Bold),
        area.width.saturating_sub(6),
    );
    frame.write_text(
        8,
        1,
        "provider setup",
        Style::default().fg(theme::CHALK_DIM),
        area.width.saturating_sub(11),
    );
}

fn draw_panel_title(frame: &mut Frame<'_>, panel: Rect, title: &str) {
    frame.write_text(
        panel.x + 2,
        panel.y,
        &format!(" {title} "),
        Style::default()
            .fg(theme::SODIUM)
            .add_modifier(Modifier::Bold),
        panel.width.saturating_sub(4),
    );
}

struct InputPanelView<'a> {
    title: &'a str,
    body: &'a [String],
    label: &'a str,
    input: &'a str,
    cursor: usize,
    error: Option<&'a str>,
    help: &'a [(&'a str, &'a str)],
}

fn draw_input_panel(frame: &mut Frame<'_>, panel: Rect, view: InputPanelView<'_>) {
    draw_panel_title(frame, panel, view.title);
    let content = inner_panel(panel);
    let mut y = content.y;

    for line in view.body {
        for wrapped in wrap_plain_text(line, content.width as usize) {
            frame.write_text(
                content.x,
                y,
                &wrapped,
                Style::default().fg(theme::CHALK_DIM),
                content.width,
            );
            y += 1;
            if y >= content.bottom() {
                return;
            }
        }
    }

    if !view.body.is_empty() {
        y += 1;
    }

    frame.write_text(
        content.x,
        y,
        view.label,
        Style::default().fg(theme::CHALK),
        content.width,
    );
    y += 1;

    let field = Rect::new(content.x, y, content.width, 3);
    frame.draw_box(
        field,
        Style::default().fg(theme::ASH),
        Some(Style::default().bg(theme::FORM)),
    );
    let inner_width = field.width.saturating_sub(4) as usize;
    let visible = visible_slice(view.input, inner_width, view.cursor);
    frame.write_text(
        field.x + 2,
        field.y + 1,
        visible,
        Style::default().fg(theme::CHALK),
        field.width.saturating_sub(4),
    );
    let vis_start = visible_start(view.input, inner_width, view.cursor);
    let cursor_char = view.input[..view.cursor]
        .chars()
        .count()
        .saturating_sub(vis_start);
    frame.set_cursor_position((field.x + 2 + cursor_char as u16, field.y + 1));
    y = field.bottom();

    if let Some(error) = view.error {
        y += 1;
        for wrapped in wrap_plain_text(error, content.width as usize) {
            frame.write_text(
                content.x,
                y,
                &wrapped,
                Style::default().fg(theme::ERROR),
                content.width,
            );
            y += 1;
            if y >= content.bottom() {
                return;
            }
        }
    }

    if y + 2 <= content.bottom() {
        frame.write_line(
            content.x,
            content.bottom().saturating_sub(1),
            &help_line(view.help),
            content.width,
        );
    }
}

fn draw_lines(frame: &mut Frame<'_>, area: Rect, lines: &[Line<'static>]) {
    for (idx, line) in lines.iter().enumerate().take(area.height as usize) {
        frame.write_line(area.x, area.y + idx as u16, line, area.width);
    }
}

fn google_auth_body(auth_url: Option<&str>, browser_error: Option<&str>) -> Vec<String> {
    let mut body = vec![
        "Authorize Lash in your browser, then paste the returned code or full callback URL here."
            .to_string(),
        format!("Manual URL is also written to {}.", oauth_url_display()),
    ];
    if auth_url.is_some() {
        if browser_error.is_some() {
            body.push("Browser open failed. Use the saved manual URL instead.".to_string());
        } else {
            body.push("A browser launch was attempted automatically.".to_string());
        }
    }
    body
}

fn help_line(items: &[(&str, &str)]) -> Line<'static> {
    let mut spans = Vec::new();
    for (idx, (key, desc)) in items.iter().enumerate() {
        if idx > 0 {
            spans.push(Span::styled("   ", Style::default().fg(theme::ASH)));
        }
        spans.push(Span::styled((*key).to_string(), theme::help_key()));
        spans.push(Span::styled(format!(" {}", desc), theme::help_desc()));
    }
    Line::from(spans)
}

fn centered_rect(area: Rect, width: u16, height: u16) -> Rect {
    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

fn inner_panel(panel: Rect) -> Rect {
    Rect::new(
        panel.x + 2,
        panel.y + 2,
        panel.width.saturating_sub(4),
        panel.height.saturating_sub(4),
    )
}

fn wrap_plain_text(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return Vec::new();
    }
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut out = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let candidate = if current.is_empty() {
            word.to_string()
        } else {
            format!("{current} {word}")
        };
        if UnicodeWidthStr::width(candidate.as_str()) <= width {
            current = candidate;
            continue;
        }
        if !current.is_empty() {
            out.push(current);
        }
        if UnicodeWidthStr::width(word) <= width {
            current = word.to_string();
        } else {
            let mut chunk = String::new();
            for ch in word.chars() {
                let candidate = format!("{chunk}{ch}");
                if UnicodeWidthStr::width(candidate.as_str()) > width && !chunk.is_empty() {
                    out.push(chunk);
                    chunk = ch.to_string();
                } else {
                    chunk.push(ch);
                }
            }
            current = chunk;
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

fn delete_left(input: &mut String, cursor: &mut usize) {
    if *cursor == 0 {
        return;
    }
    let previous = input[..*cursor]
        .char_indices()
        .last()
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    input.drain(previous..*cursor);
    *cursor = previous;
}

fn move_left(input: &str, cursor: &mut usize) {
    if *cursor == 0 {
        return;
    }
    *cursor = input[..*cursor]
        .char_indices()
        .last()
        .map(|(idx, _)| idx)
        .unwrap_or(0);
}

fn move_right(input: &str, cursor: &mut usize) {
    if *cursor >= input.len() {
        return;
    }
    *cursor += input[*cursor..]
        .chars()
        .next()
        .map(|ch| ch.len_utf8())
        .unwrap_or(0);
}

fn insert_text(input: &mut String, cursor: &mut usize, text: &str) {
    if text.is_empty() {
        return;
    }
    input.insert_str(*cursor, text);
    *cursor += text.len();
}

fn visible_slice(input: &str, width: usize, cursor_byte: usize) -> &str {
    if width == 0 {
        return "";
    }
    let total = input.chars().count();
    if total <= width {
        return input;
    }
    let chars: Vec<(usize, char)> = input.char_indices().collect();
    let cursor_char = input[..cursor_byte].chars().count();
    let start = cursor_char
        .saturating_sub(width)
        .min(total.saturating_sub(width));
    let end = (start + width).min(total);
    let byte_start = chars[start].0;
    let byte_end = if end < total {
        chars[end].0
    } else {
        input.len()
    };
    &input[byte_start..byte_end]
}

fn visible_start(input: &str, width: usize, cursor_byte: usize) -> usize {
    let total = input.chars().count();
    if total <= width {
        return 0;
    }
    let cursor_char = input[..cursor_byte].chars().count();
    cursor_char
        .saturating_sub(width)
        .min(total.saturating_sub(width))
}

fn existing_openai_base_url(existing: Option<&LashConfig>) -> String {
    existing
        .and_then(|cfg| cfg.provider(ProviderKind::OpenAiGeneric))
        .and_then(|provider| match provider {
            Provider::OpenAiGeneric { base_url, .. } => Some(base_url.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

fn existing_openai_key(existing: Option<&LashConfig>) -> String {
    existing
        .and_then(|cfg| cfg.provider(ProviderKind::OpenAiGeneric))
        .and_then(|provider| match provider {
            Provider::OpenAiGeneric { api_key, .. } => Some(api_key.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

fn open_browser(url: &str) -> std::io::Result<()> {
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
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;
    }
    Ok(())
}

fn copy_to_clipboard(text: &str) -> Result<(), String> {
    crate::clipboard::copy_text_robustly(text)
        .map(|_| ())
        .map_err(|err| format!("Failed to copy code: {err}"))
}

fn oauth_url_path() -> std::path::PathBuf {
    let mut path = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    path.push(".lash");
    path.push("oauth-url.txt");
    path
}

fn oauth_url_display() -> String {
    if let Some(home) = std::env::var_os("HOME").map(std::path::PathBuf::from) {
        let full = oauth_url_path();
        if let Ok(relative) = full.strip_prefix(&home) {
            return format!("~/{}", relative.display());
        }
    }
    oauth_url_path().display().to_string()
}

fn persist_oauth_url(url: &str) {
    let path = oauth_url_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, url);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_text_without_dropping_words() {
        let lines = wrap_plain_text("alpha beta gamma delta", 10);
        assert!(!lines.is_empty());
        assert_eq!(lines.join(" "), "alpha beta gamma delta");
    }

    #[test]
    fn visible_slice_tracks_cursor_tail() {
        let input = "abcdefghijklmnopqrstuvwxyz";
        assert_eq!(visible_slice(input, 6, input.len()), "uvwxyz");
    }
}
