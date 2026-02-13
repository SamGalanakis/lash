use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

use crate::app::{App, DisplayBlock};

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
        Span::styled(" kaml", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(" \u{2500} ", Style::default().fg(Color::DarkGray)),
        Span::styled(&app.model, Style::default().fg(Color::White)),
    ];

    if app.running {
        let spinner_char = SPINNER[app.tick % SPINNER.len()];
        spans.push(Span::styled(
            format!("  {}", spinner_char),
            Style::default().fg(Color::Yellow),
        ));
        if let Some(status) = &app.status_text {
            spans.push(Span::styled(
                format!(" {}", status),
                Style::default().fg(Color::DarkGray),
            ));
        }
    }

    let bar = Paragraph::new(Line::from(spans))
        .style(Style::default().bg(Color::Rgb(30, 30, 40)));
    frame.render_widget(bar, area);
}

fn draw_history(frame: &mut Frame, app: &App, area: Rect) {
    let viewport_height = area.height as usize;
    let total_height = app.total_content_height();

    // Clamp scroll offset
    let max_scroll = total_height.saturating_sub(viewport_height);
    let scroll = app.scroll_offset.min(max_scroll);

    let mut lines: Vec<Line> = Vec::new();

    for block in &app.blocks {
        render_block(block, app.code_expanded, &mut lines);
    }

    // Pad to fill viewport if content is shorter
    while lines.len() < viewport_height {
        lines.push(Line::from(""));
    }

    let paragraph = Paragraph::new(lines)
        .scroll((scroll as u16, 0))
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::NONE));

    frame.render_widget(paragraph, area);
}

fn render_block<'a>(block: &'a DisplayBlock, code_expanded_global: bool, lines: &mut Vec<Line<'a>>) {
    match block {
        DisplayBlock::UserInput(text) => {
            for line in text.lines() {
                lines.push(Line::from(vec![
                    Span::styled("> ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                    Span::styled(line.to_string(), Style::default().fg(Color::Green)),
                ]));
            }
            lines.push(Line::from(""));
        }
        DisplayBlock::AssistantText(text) => {
            for line in text.lines() {
                lines.push(Line::from(Span::raw(line.to_string())));
            }
        }
        DisplayBlock::CodeBlock { code, expanded } => {
            let show = code_expanded_global && *expanded;
            let line_count = code.lines().count();
            if show {
                lines.push(Line::from(Span::styled(
                    format!("\u{25bc} python ({} lines)", line_count),
                    Style::default().fg(Color::DarkGray),
                )));
                for line in code.lines() {
                    lines.push(Line::from(vec![
                        Span::styled("\u{2502} ", Style::default().fg(Color::DarkGray)),
                        Span::styled(line.to_string(), Style::default().fg(Color::Rgb(180, 180, 220))),
                    ]));
                }
                lines.push(Line::from(Span::styled(
                    "\u{2514}\u{2500}\u{2500}\u{2500}",
                    Style::default().fg(Color::DarkGray),
                )));
            } else {
                lines.push(Line::from(Span::styled(
                    format!("\u{25b6} python ({} lines)", line_count),
                    Style::default().fg(Color::DarkGray),
                )));
            }
        }
        DisplayBlock::ToolCall {
            name,
            success,
            duration_ms,
        } => {
            let icon = if *success { "+" } else { "x" };
            let color = if *success { Color::DarkGray } else { Color::Red };
            lines.push(Line::from(Span::styled(
                format!("  [{}] {} ({}ms)", icon, name, duration_ms),
                Style::default().fg(color),
            )));
        }
        DisplayBlock::CodeOutput { output, error } => {
            // stdout is the model's debug buffer — only show when expanded (CTRL+O)
            let show_stdout = code_expanded_global && !output.is_empty();
            let has_error = error.is_some();

            if show_stdout {
                lines.push(Line::from(Span::styled(
                    "\u{251c}\u{2500} stdout \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}",
                    Style::default().fg(Color::DarkGray),
                )));
                for line in output.lines() {
                    lines.push(Line::from(vec![
                        Span::styled("\u{2502} ", Style::default().fg(Color::DarkGray)),
                        Span::styled(line.to_string(), Style::default().fg(Color::DarkGray)),
                    ]));
                }
            }
            if let Some(err) = error {
                lines.push(Line::from(vec![
                    Span::styled("\u{251c}\u{2500} ", Style::default().fg(Color::DarkGray)),
                    Span::styled("error", Style::default().fg(Color::Red)),
                    Span::styled(
                        " \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}",
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
                for line in err.lines() {
                    lines.push(Line::from(vec![
                        Span::styled("\u{2502} ", Style::default().fg(Color::DarkGray)),
                        Span::styled(line.to_string(), Style::default().fg(Color::Red)),
                    ]));
                }
            }
            if show_stdout || has_error {
                lines.push(Line::from(Span::styled(
                    "\u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}",
                    Style::default().fg(Color::DarkGray),
                )));
            }
        }
        DisplayBlock::Error(msg) => {
            lines.push(Line::from(Span::styled(
                format!("Error: {}", msg),
                Style::default().fg(Color::Red),
            )));
        }
    }
}

fn draw_input(frame: &mut Frame, app: &App, area: Rect) {
    let display = if app.running {
        "  waiting for agent...".to_string()
    } else {
        format!("> {}", &app.input)
    };

    let style = if app.running {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(Color::White)
    };

    let input = Paragraph::new(display)
        .style(style)
        .block(
            Block::default()
                .borders(Borders::TOP | Borders::BOTTOM)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
    frame.render_widget(input, area);

    // Position cursor
    if !app.running {
        // ">" + " " = 2 chars offset
        let cursor_x = area.x + 2 + visual_width(&app.input[..app.cursor_pos]) as u16;
        let cursor_y = area.y + 1; // inside the border
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}

fn draw_help_bar(frame: &mut Frame, _app: &App, area: Rect) {
    let help = Line::from(vec![
        Span::styled(" ^O", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(" toggle code  ", Style::default().fg(Color::DarkGray)),
        Span::styled("^C", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(" quit  ", Style::default().fg(Color::DarkGray)),
        Span::styled("PgUp/PgDn", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(" scroll  ", Style::default().fg(Color::DarkGray)),
        Span::styled("\u{2191}\u{2193}", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(" history", Style::default().fg(Color::DarkGray)),
    ]);

    let bar = Paragraph::new(help).style(Style::default().bg(Color::Rgb(30, 30, 40)));
    frame.render_widget(bar, area);
}

fn visual_width(s: &str) -> usize {
    // Simple ASCII width — good enough for input line
    s.chars().count()
}
