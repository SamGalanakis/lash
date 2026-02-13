use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

use crate::app::{App, DisplayBlock};
use crate::theme;

const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

pub fn draw(frame: &mut Frame, app: &App) {
    let chunks = Layout::vertical([
        Constraint::Length(1), // status bar
        Constraint::Min(3),   // history
        Constraint::Length(3), // input
        Constraint::Length(1), // help bar
    ])
    .split(frame.area());

    draw_status_bar(frame, app, chunks[0]);
    draw_history(frame, app, chunks[1]);
    draw_input(frame, app, chunks[2]);
    draw_help_bar(frame, app, chunks[3]);
}

fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let mut spans = vec![
        Span::styled(" lash", theme::app_title()),
        Span::styled(theme::STATUS_SEP, theme::status_separator()),
        Span::styled(&app.model, theme::model_name()),
    ];

    if app.running {
        let spinner_char = SPINNER[app.tick % SPINNER.len()];
        spans.push(Span::styled(
            format!("  {}", spinner_char),
            theme::spinner(),
        ));
        if let Some(status) = &app.status_text {
            spans.push(Span::styled(
                format!(" {}", status),
                theme::status_text(),
            ));
        }
    }

    let bar = Paragraph::new(Line::from(spans)).style(theme::bar_bg());
    frame.render_widget(bar, area);
}

fn draw_history(frame: &mut Frame, app: &App, area: Rect) {
    let viewport_height = area.height as usize;
    let viewport_width = area.width as usize;
    let total_height = app.total_content_height(viewport_width);

    // Clamp scroll offset
    let max_scroll = total_height.saturating_sub(viewport_height);
    let scroll = app.scroll_offset.min(max_scroll);

    let mut lines: Vec<Line> = Vec::new();

    for block in &app.blocks {
        render_block(block, app.code_expanded, &mut lines, viewport_width, viewport_height);
    }

    // Pad to fill viewport if content is shorter
    while lines.len() < viewport_height {
        lines.push(Line::from(""));
    }

    let paragraph = Paragraph::new(lines)
        .scroll((scroll as u16, 0))
        .wrap(Wrap { trim: false })
        .style(theme::history_bg())
        .block(Block::default().borders(Borders::NONE));

    frame.render_widget(paragraph, area);
}

fn render_block<'a>(
    block: &'a DisplayBlock,
    code_expanded_global: bool,
    lines: &mut Vec<Line<'a>>,
    viewport_width: usize,
    viewport_height: usize,
) {
    match block {
        DisplayBlock::UserInput(text) => {
            for line in text.lines() {
                lines.push(Line::from(vec![
                    Span::styled(format!("{} ", theme::PROMPT_CHAR), theme::prompt()),
                    Span::styled(line.to_string(), theme::user_input()),
                ]));
            }
            lines.push(Line::from(""));
        }
        DisplayBlock::AssistantText(text) => {
            for line in text.lines() {
                lines.push(Line::from(Span::styled(line.to_string(), theme::assistant_text())));
            }
        }
        DisplayBlock::CodeBlock { code, expanded } => {
            let show = code_expanded_global && *expanded;
            let line_count = code.lines().count();
            if show {
                lines.push(Line::from(Span::styled(
                    format!("\u{25bc} python ({} lines)", line_count),
                    theme::code_header(),
                )));
                for line in code.lines() {
                    lines.push(Line::from(vec![
                        Span::styled("\u{2502} ", theme::code_chrome()),
                        Span::styled(line.to_string(), theme::code_content()),
                    ]));
                }
                lines.push(Line::from(Span::styled(
                    "\u{2514}\u{2500}\u{2500}\u{2500}",
                    theme::code_chrome(),
                )));
            } else {
                lines.push(Line::from(Span::styled(
                    format!("\u{25b6} python ({} lines)", line_count),
                    theme::code_header(),
                )));
            }
        }
        DisplayBlock::ToolCall {
            name,
            success,
            duration_ms,
        } => {
            let icon = if *success { "+" } else { "x" };
            let style = if *success { theme::tool_success() } else { theme::tool_failure() };
            lines.push(Line::from(Span::styled(
                format!("  [{}] {} ({}ms)", icon, name, duration_ms),
                style,
            )));
        }
        DisplayBlock::CodeOutput { output, error } => {
            // stdout is the model's debug buffer — only show when expanded (CTRL+O)
            let show_stdout = code_expanded_global && !output.is_empty();
            let has_error = error.is_some();

            if show_stdout {
                lines.push(Line::from(Span::styled(
                    "\u{251c}\u{2500} stdout \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}",
                    theme::code_chrome(),
                )));
                for line in output.lines() {
                    lines.push(Line::from(vec![
                        Span::styled("\u{2502} ", theme::code_chrome()),
                        Span::styled(line.to_string(), theme::system_output()),
                    ]));
                }
            }
            if let Some(err) = error {
                lines.push(Line::from(vec![
                    Span::styled("\u{251c}\u{2500} ", theme::code_chrome()),
                    Span::styled("error", theme::error()),
                    Span::styled(
                        " \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}",
                        theme::code_chrome(),
                    ),
                ]));
                for line in err.lines() {
                    lines.push(Line::from(vec![
                        Span::styled("\u{2502} ", theme::code_chrome()),
                        Span::styled(line.to_string(), theme::error()),
                    ]));
                }
            }
            if show_stdout || has_error {
                lines.push(Line::from(Span::styled(
                    "\u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}",
                    theme::code_chrome(),
                )));
            }
        }
        DisplayBlock::Error(msg) => {
            lines.push(Line::from(Span::styled(
                format!("Error: {}", msg),
                theme::error(),
            )));
        }
        DisplayBlock::Splash => {
            use ratatui::style::Style;

            let chalk = theme::assistant_text();
            let sodium = Style::default().fg(theme::SODIUM);

            // LASH wordmark — thick sodium ██ slash overlays letters
            // Letters are complete; slash recolors blocks where it crosses
            //
            //  ██       ████   ██████▓▓  ██
            //  ██      ██  ██  ██   ▓▓█  ██
            //  ██      ██████  ████▓▓██████
            //  ██      ██  ██      ▓▓██  ██
            //  ██████  ██  ██  ██▓▓  ██  ██
            //                    ▓▓
            let content_width = 30;
            let content_height = 9; // 6 logo + scribe + tagline + blank
            let cx = viewport_width.saturating_sub(content_width) / 2;
            let cy = viewport_height.saturating_sub(content_height) / 2;
            let pad = " ".repeat(cx);

            for _ in 0..cy {
                lines.push(Line::from(""));
            }

            // Each row: [before chalk] [██ sodium overlay] [after chalk]
            let logo: &[(&str, &str)] = &[
                ("██       ████   ██████  ", "  ██"),
                ("██      ██  ██  ██     ", "█  ██"),
                ("██      ██████  ██████", "██████"),
                ("██      ██  ██      █", " ██  ██"),
                ("██████  ██  ██  ████", "  ██  ██"),
            ];
            for &(before, after) in logo {
                lines.push(Line::from(vec![
                    Span::styled(format!("{}{}", pad, before), chalk),
                    Span::styled("██", sodium),
                    Span::styled(after, chalk),
                ]));
            }
            // Slash tail
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
            lines.push(Line::from(""));
        }
    }
}

fn draw_input(frame: &mut Frame, app: &App, area: Rect) {
    let content = if app.running {
        Line::from(Span::styled("  waiting for agent...", theme::waiting()))
    } else {
        Line::from(vec![
            Span::styled(format!("{} ", theme::PROMPT_CHAR), theme::prompt()),
            Span::styled(&app.input, theme::user_input()),
        ])
    };

    let input = Paragraph::new(content)
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::TOP | Borders::BOTTOM)
                .border_style(theme::input_border()),
        );
    frame.render_widget(input, area);

    // Position cursor
    if !app.running {
        // "/" + " " = 2 chars offset
        let cursor_x = area.x + 2 + visual_width(&app.input[..app.cursor_pos]) as u16;
        let cursor_y = area.y + 1; // inside the border
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}

fn draw_help_bar(frame: &mut Frame, _app: &App, area: Rect) {
    let help = Line::from(vec![
        Span::styled(" ^O", theme::help_key()),
        Span::styled(" toggle code  ", theme::help_desc()),
        Span::styled("^C", theme::help_key()),
        Span::styled(" quit  ", theme::help_desc()),
        Span::styled("PgUp/PgDn", theme::help_key()),
        Span::styled(" scroll  ", theme::help_desc()),
        Span::styled("\u{2191}\u{2193}", theme::help_key()),
        Span::styled(" history", theme::help_desc()),
    ]);

    let bar = Paragraph::new(help).style(theme::bar_bg());
    frame.render_widget(bar, area);
}

fn visual_width(s: &str) -> usize {
    // Simple ASCII width — good enough for input line
    s.chars().count()
}
