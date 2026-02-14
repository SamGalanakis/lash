use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};
use unicode_width::UnicodeWidthStr;

use crate::app::{App, DisplayBlock, SuggestionKind};
use crate::markdown;
use crate::theme;

pub fn draw(frame: &mut Frame, app: &App) {
    // Paint entire frame with FORM bg so no terminal background bleeds through
    frame.render_widget(Block::default().style(theme::history_bg()), frame.area());

    let strike_h = if app.running { 1 } else { 0 };
    let image_h = if app.pending_images.is_empty() { 0 } else { 1 };

    // Dynamic input height: account for visual wrapping, not just newline count.
    // Inner width = frame width minus 2 (border chars are 0 since TOP/BOTTOM only, but
    // the "/ " prefix eats 2 columns).
    let inner_w = frame.area().width.saturating_sub(2) as usize; // usable columns after "/ " prefix
    let visual_lines = input_visual_lines(&app.input, inner_w);
    let queue_lines = app.queue_count() as u16;
    let input_h = (visual_lines as u16 + 2 + queue_lines).min(12); // +2 for borders, max 12 for queue room

    let chunks = Layout::vertical([
        Constraint::Length(1),        // status bar
        Constraint::Min(3),           // history
        Constraint::Length(strike_h), // strike zone (only when running)
        Constraint::Length(image_h),  // image attachment badges
        Constraint::Length(input_h),  // input (dynamic height)
        Constraint::Length(1),        // help bar
    ])
    .split(frame.area());

    draw_status_bar(frame, app, chunks[0]);
    draw_history(frame, app, chunks[1]);
    if app.running {
        draw_strike_zone(frame, app, chunks[2]);
    }
    draw_image_badges(frame, app, chunks[3]);
    draw_input(frame, app, chunks[4]);
    draw_suggestions(frame, app, chunks[4]);
    draw_session_picker(frame, app, chunks[1]); // overlay on history area
    draw_skill_picker(frame, app, chunks[1]);   // overlay on history area
    draw_prompt(frame, app, chunks[1]);          // overlay on history area
    draw_help_bar(frame, app, chunks[5]);
}

fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let mut left_spans = vec![
        Span::styled(" lash", theme::app_title()),
        Span::styled(theme::STATUS_SEP, theme::status_separator()),
        Span::styled(&app.model, theme::model_name()),
    ];

    if app.mode == crate::app::Mode::Plan {
        left_spans.push(Span::styled(theme::STATUS_SEP, theme::status_separator()));
        left_spans.push(Span::styled(
            "PLAN",
            Style::default().fg(theme::SODIUM).add_modifier(Modifier::BOLD),
        ));
    }

    // Token counts on the right side (only if we have usage data)
    if app.token_usage.total() > 0 {
        let mut token_str = format!(
            "{} in \u{b7} {} out \u{b7} {} total",
            crate::app::format_tokens(app.token_usage.input_tokens),
            crate::app::format_tokens(app.token_usage.output_tokens),
            crate::app::format_tokens(app.token_usage.total()),
        );

        // Append context window percentage if available
        let ctx_pct = if let Some(ctx_win) = app.context_window {
            if app.last_input_tokens > 0 && ctx_win > 0 {
                Some((app.last_input_tokens as f64 / ctx_win as f64 * 100.0) as u64)
            } else {
                None
            }
        } else {
            None
        };
        if let Some(pct) = ctx_pct {
            token_str.push_str(&format!(" \u{b7} {}% ctx", pct));
        }

        // Calculate padding to right-align
        let left_width: usize = left_spans.iter().map(|s| s.content.chars().count()).sum();
        let right_width = token_str.chars().count() + 1; // +1 trailing space
        let pad = (area.width as usize).saturating_sub(left_width + right_width);
        if pad > 0 {
            left_spans.push(Span::styled(" ".repeat(pad), theme::bar_bg()));
        }

        // Split token string and context percentage into separate styled spans
        if let Some(pct) = ctx_pct {
            let ctx_suffix = format!(" \u{b7} {}% ctx", pct);
            let tokens_part = &token_str[..token_str.len() - ctx_suffix.len()];
            left_spans.push(Span::styled(tokens_part.to_string(), Style::default().fg(theme::CHALK_DIM)));
            let ctx_color = if pct >= 80 { theme::SODIUM } else { theme::CHALK_DIM };
            left_spans.push(Span::styled(ctx_suffix, Style::default().fg(ctx_color)));
        } else {
            left_spans.push(Span::styled(token_str, Style::default().fg(theme::CHALK_DIM)));
        }
    }

    let bar = Paragraph::new(Line::from(left_spans)).style(theme::bar_bg());
    frame.render_widget(bar, area);
}

fn draw_history(frame: &mut Frame, app: &App, area: Rect) {
    let viewport_height = area.height as usize;
    let viewport_width = area.width as usize;

    // scroll_offset is already clamped by the main loop before draw()
    let scroll = app.scroll_offset;

    // Find the first visible block via binary search on the height cache
    let (first_idx, skip_lines) = find_visible_block_readonly(app, scroll, viewport_width);

    let mut lines: Vec<Line> = Vec::with_capacity(viewport_height + skip_lines + 20);

    // Render only blocks from first_idx until we have enough lines
    for block in app.blocks[first_idx..].iter() {
        render_block(block, app.code_expanded, &mut lines, viewport_width, viewport_height);
        if lines.len() >= viewport_height + skip_lines {
            break;
        }
    }

    // Render streaming pending_text (code lines only, when code_expanded)
    if app.running && app.code_expanded && !app.pending_text.is_empty() && lines.len() < viewport_height + skip_lines {
        for line in app.pending_text.lines() {
            let trimmed = line.trim_start();
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                lines.push(Line::from(Span::styled(
                    format!("  {}", line),
                    theme::code_content(),
                )));
            }
            if lines.len() >= viewport_height + skip_lines {
                break;
            }
        }
    }

    // Render live streaming tool output (e.g. bash)
    if !app.streaming_output.is_empty() && lines.len() < viewport_height + skip_lines {
        for line in &app.streaming_output {
            lines.push(Line::from(vec![
                Span::styled("\u{2502} ", theme::code_chrome()),
                Span::styled(line.as_str(), theme::code_content()),
            ]));
            if lines.len() >= viewport_height + skip_lines {
                break;
            }
        }
    }

    // Bottom-align content when it doesn't fill the viewport (chat-style).
    // Use visual row count (accounting for line wrapping) not logical line count.
    if scroll == 0 {
        let total_visual: usize = lines.iter().map(|line| {
            let w = line.width();
            if w == 0 { 1 } else { (w + viewport_width - 1) / viewport_width }
        }).sum();
        let pad_count = viewport_height.saturating_sub(total_visual);
        if pad_count > 0 {
            let mut padded = Vec::with_capacity(pad_count + lines.len());
            for _ in 0..pad_count {
                padded.push(Line::from(""));
            }
            padded.extend(lines);
            lines = padded;
        }
    }

    // Use Paragraph's built-in scroll to skip visual rows correctly.
    // Manual .skip()/.take() on logical Lines is wrong because skip_lines
    // is in visual-row space (from the height cache) but Lines can wrap.
    let paragraph = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .style(theme::history_bg())
        .block(Block::default().borders(Borders::NONE))
        .scroll((skip_lines as u16, 0));

    frame.render_widget(paragraph, area);
}

/// Read-only version of find_visible_block that uses the pre-computed height cache.
/// The cache MUST be pre-warmed via `ensure_height_cache_pub` before calling draw().
fn find_visible_block_readonly(app: &App, scroll_offset: usize, _width: usize) -> (usize, usize) {
    if app.height_cache_snapshot().is_empty() {
        return (0, 0);
    }
    let cache = app.height_cache_snapshot();
    let idx = cache.partition_point(|&cumulative| cumulative <= scroll_offset);
    if idx >= app.blocks.len() {
        return (app.blocks.len(), 0);
    }
    let block_start = if idx == 0 { 0 } else { cache[idx - 1] };
    let skip = scroll_offset - block_start;
    (idx, skip)
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
                    Span::styled(line, theme::user_input()),
                ]));
            }
            lines.push(Line::from(""));
        }
        DisplayBlock::AssistantText(text) => {
            let rendered = markdown::render_markdown(text);
            lines.extend(rendered);
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
                        Span::styled(line, theme::code_content()),
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
                        Span::styled(line, theme::system_output()),
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
                        Span::styled(line, theme::error()),
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
        DisplayBlock::ShellOutput { command, output, error } => {
            lines.push(Line::from(Span::styled(
                format!("$ {}", command),
                theme::code_chrome(),
            )));
            if !output.is_empty() {
                for line in output.lines() {
                    lines.push(Line::from(vec![
                        Span::styled("\u{2502} ", theme::code_chrome()),
                        Span::styled(line, theme::system_output()),
                    ]));
                }
            }
            if let Some(err) = error {
                for line in err.lines() {
                    lines.push(Line::from(vec![
                        Span::styled("\u{2502} ", theme::code_chrome()),
                        Span::styled(line, theme::error()),
                    ]));
                }
            }
            lines.push(Line::from(Span::styled(
                "\u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}",
                theme::code_chrome(),
            )));
        }
        DisplayBlock::Error(msg) => {
            lines.push(Line::from(Span::styled(
                format!("Error: {}", msg),
                theme::error(),
            )));
        }
        DisplayBlock::SystemMessage(text) => {
            for line in text.lines() {
                lines.push(Line::from(Span::styled(line, theme::system_message())));
            }
            lines.push(Line::from(""));
        }
        DisplayBlock::SubAgentResult {
            task,
            usage,
            tool_calls,
            success,
            is_last,
            ..
        } => {
            let connector = if *is_last { "\u{2514}\u{2500}" } else { "\u{251c}\u{2500}" };
            let status_connector = if *is_last { "   " } else { "\u{2502}  " };

            // Truncate task to fit: connector(2) + space(1) + task + sep(3) + stats(~30)
            let stats_str = format!(
                " \u{b7} {} tool uses \u{b7} {} tokens",
                tool_calls,
                crate::app::format_tokens(usage.total()),
            );
            let max_task_w = viewport_width.saturating_sub(3 + stats_str.chars().count());
            let task_display: String = task.chars().take(max_task_w).collect();
            let truncated = task.chars().count() > max_task_w;
            let task_final = if truncated {
                format!("{}\u{2026}", task_display)
            } else {
                task_display
            };

            lines.push(Line::from(vec![
                Span::styled(format!("{} ", connector), Style::default().fg(theme::ASH)),
                Span::styled(task_final, Style::default().fg(theme::CHALK_MID)),
                Span::styled(stats_str, Style::default().fg(theme::CHALK_DIM)),
            ]));

            let (status_label, status_style) = if *success {
                ("\u{2514}\u{2500} Done", Style::default().fg(theme::LICHEN))
            } else {
                ("\u{2514}\u{2500} Failed", Style::default().fg(theme::ERROR))
            };
            lines.push(Line::from(vec![
                Span::styled(status_connector, Style::default().fg(theme::ASH)),
                Span::styled(status_label, status_style),
            ]));
        }
        DisplayBlock::PlanContent(content) => {
            render_plan_block(content, lines, viewport_width);
        }
        DisplayBlock::Splash => {
            let chalk = theme::assistant_text();
            let sodium = Style::default().fg(theme::SODIUM);

            let content_width = 30;
            let content_height = 9;
            let cx = viewport_width.saturating_sub(content_width) / 2;
            let cy = viewport_height.saturating_sub(content_height) / 2;
            let pad = " ".repeat(cx);

            for _ in 0..cy {
                lines.push(Line::from(""));
            }

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
            lines.push(Line::from(Span::styled(
                format!("{}                   ██", pad),
                sodium,
            )));
            lines.push(Line::from(vec![
                Span::styled(format!("{}──────────", pad), sodium),
                Span::styled("──────────", Style::default().fg(theme::ASH_MID)),
                Span::styled("──────────", Style::default().fg(theme::ASH)),
            ]));
            lines.push(Line::from(Span::styled(
                format!("{}A G E N T  \u{b7}  R U N T I M E", pad),
                Style::default().fg(theme::ASH_TEXT),
            )));
            lines.push(Line::from(""));

            // Bottom padding to fill viewport (so bottom-align logic sees full viewport)
            let rendered = cy + content_height;
            for _ in rendered..viewport_height {
                lines.push(Line::from(""));
            }
        }
    }
}

/// Render plan content as a bordered block with markdown inside.
fn render_plan_block<'a>(content: &'a str, lines: &mut Vec<Line<'a>>, viewport_width: usize) {
    let title = " PLAN ";
    let border_w = viewport_width.saturating_sub(2); // inside left/right border chars
    let title_len = title.len();
    let rule_after = border_w.saturating_sub(title_len + 1); // +1 for opening corner

    // Top border: ┌─ PLAN ─────────────────────────┐
    let top = format!(
        "\u{250c}\u{2500}{}{}",
        title,
        "\u{2500}".repeat(rule_after.saturating_sub(1))
    );
    // Ensure we close with ┐ if there's room
    let top_line = if viewport_width > 2 {
        let padded = format!("{}\u{2510}", &top[..top.chars().count().min(viewport_width - 1)]);
        padded
    } else {
        top
    };

    lines.push(Line::from(vec![
        Span::styled(
            top_line.chars().take(2).collect::<String>(), // ┌─
            Style::default().fg(theme::ASH),
        ),
        Span::styled(
            title.to_string(),
            Style::default().fg(theme::SODIUM).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(
                "{}\u{2510}",
                "\u{2500}".repeat(rule_after.saturating_sub(1))
            ),
            Style::default().fg(theme::ASH),
        ),
    ]));

    // Content: render markdown lines with │ prefix
    let md_lines = markdown::render_markdown(content);
    for line in md_lines {
        let mut spans = vec![Span::styled("\u{2502} ", Style::default().fg(theme::ASH))];
        spans.extend(line.spans);
        lines.push(Line::from(spans));
    }

    // Bottom border: └──────────────────────────────┘
    let bottom_fill = viewport_width.saturating_sub(2); // minus └ and ┘
    lines.push(Line::from(Span::styled(
        format!(
            "\u{2514}{}\u{2518}",
            "\u{2500}".repeat(bottom_fill)
        ),
        Style::default().fg(theme::ASH),
    )));
}

// ── Strike zone ─────────────────────────────────────────────────────
// The brand slash rotating in place — a tight 2-char animation where
// the lead character glows sodium and the trail fades to ash. Status
// text follows on the same line. One line, focused, no noise.

fn draw_strike_zone(frame: &mut Frame, app: &App, area: Rect) {
    // Four angles of the slash mark rotating in place
    const ANGLES: &[char] = &['╲', '─', '╱', '│'];
    // Divide tick by 2 → 200ms per angle, 800ms full rotation
    let idx = (app.tick / 2) % ANGLES.len();
    let trail_idx = (idx + ANGLES.len() - 1) % ANGLES.len();

    let lead = ANGLES[idx];
    let trail = ANGLES[trail_idx];

    let status = app.status_text.as_deref().unwrap_or("");

    let mut spans = vec![
        Span::raw("  "),
        Span::styled(
            trail.to_string(),
            Style::default().fg(theme::ASH_TEXT),
        ),
        Span::styled(
            lead.to_string(),
            Style::default().fg(theme::SODIUM).add_modifier(Modifier::BOLD),
        ),
    ];

    if !status.is_empty() {
        spans.push(Span::styled(
            format!("  {}", status),
            Style::default().fg(theme::ASH_MID),
        ));
    }

    let paragraph = Paragraph::new(Line::from(spans)).style(theme::history_bg());
    frame.render_widget(paragraph, area);
}

fn draw_image_badges(frame: &mut Frame, app: &App, area: Rect) {
    if app.pending_images.is_empty() {
        return;
    }

    let badges: Vec<Span> = app
        .pending_images
        .iter()
        .enumerate()
        .flat_map(|(i, _)| {
            let mut spans = Vec::new();
            if i > 0 {
                spans.push(Span::raw("  "));
            }
            spans.push(Span::styled(
                format!(" \u{1F5BC} Image {} ", i + 1),
                theme::image_attachment(),
            ));
            spans
        })
        .collect();

    // Left padding to align with the prompt character
    let mut spans = vec![Span::raw("  ")];
    spans.extend(badges);
    let paragraph = Paragraph::new(Line::from(spans)).style(theme::history_bg());
    frame.render_widget(paragraph, area);
}

fn draw_input(frame: &mut Frame, app: &App, area: Rect) {
    let mut lines = Vec::new();

    // Render queued messages above the prompt
    let queue_len = app.message_queue.len();
    if queue_len > 0 {
        let max_msg_w = area.width.saturating_sub(6) as usize; // "  N. " prefix width
        for (i, msg) in app.message_queue.iter().enumerate() {
            let num = i + 1;
            let is_last = i == queue_len - 1;
            // Collapse newlines to spaces and truncate
            let collapsed: String = msg.chars().map(|c| if c == '\n' { ' ' } else { c }).collect();
            let truncated: String = collapsed.chars().take(max_msg_w).collect();
            let display = if collapsed.chars().count() > max_msg_w {
                format!("{}\u{2026}", truncated)
            } else {
                truncated
            };
            let content_style = if is_last { theme::CHALK_MID } else { theme::CHALK_DIM };
            lines.push(Line::from(vec![
                Span::styled(format!("  {}. ", num), Style::default().fg(theme::ASH_MID)),
                Span::styled(display, Style::default().fg(content_style)),
            ]));
        }
    }

    // Multi-line input rendering
    let input_lines: Vec<&str> = app.input.split('\n').collect();
    for (i, line) in input_lines.iter().enumerate() {
        if i == 0 {
            lines.push(Line::from(vec![
                Span::styled(format!("{} ", theme::PROMPT_CHAR), theme::prompt()),
                Span::styled(line.to_string(), theme::user_input()),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::styled("  ", Style::default().fg(theme::ASH)), // continuation indent
                Span::styled(line.to_string(), theme::user_input()),
            ]));
        }
    }

    let input = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::TOP | Borders::BOTTOM)
                .border_style(theme::input_border()),
        );
    frame.render_widget(input, area);

    // Position cursor accounting for visual wrapping and queue lines
    let (vis_row, vis_col) = input_cursor_position(&app.input, app.cursor_pos, area.width as usize);
    let extra_offset = queue_len as u16; // lines rendered above input text

    // Scroll: if cursor is past the visible content area, scroll the paragraph
    let content_h = area.height.saturating_sub(2) as usize; // inside borders
    if vis_row + (extra_offset as usize) < content_h {
        let cursor_x = area.x + vis_col as u16;
        let cursor_y = area.y + 1 + extra_offset + vis_row as u16;
        frame.set_cursor_position((cursor_x, cursor_y));
    } else {
        // Cursor beyond visible area — place at bottom-right as indicator
        let cursor_x = area.x + vis_col as u16;
        let cursor_y = area.y + area.height.saturating_sub(2);
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}

fn draw_help_bar(frame: &mut Frame, app: &App, area: Rect) {
    let help = if app.running {
        let mut spans = vec![
            Span::styled(" Esc", theme::help_key()),
            Span::styled(" cancel  ", theme::help_desc()),
            Span::styled("Enter", theme::help_key()),
            Span::styled(" queue  ", theme::help_desc()),
            Span::styled("^O", theme::help_key()),
            Span::styled(" toggle code  ", theme::help_desc()),
        ];
        if app.queue_count() > 0 {
            spans.push(Span::styled("Bksp", theme::help_key()));
            spans.push(Span::styled(" unqueue  ", theme::help_desc()));
            spans.push(Span::styled(
                format!(" {} queued ", app.queue_count()),
                Style::default().fg(theme::SODIUM).add_modifier(Modifier::BOLD),
            ));
        }
        spans.push(Span::styled("^C", theme::help_key()));
        spans.push(Span::styled(" quit", theme::help_desc()));
        Line::from(spans)
    } else {
        Line::from(vec![
            Span::styled(" ^U/^D", theme::help_key()),
            Span::styled(" scroll  ", theme::help_desc()),
            Span::styled("S-Tab", theme::help_key()),
            Span::styled(" plan  ", theme::help_desc()),
            Span::styled("^O", theme::help_key()),
            Span::styled(" code  ", theme::help_desc()),
            Span::styled("S-Enter", theme::help_key()),
            Span::styled(" newline  ", theme::help_desc()),
            Span::styled("^Y", theme::help_key()),
            Span::styled(" copy  ", theme::help_desc()),
            Span::styled("^C", theme::help_key()),
            Span::styled(" quit", theme::help_desc()),
        ])
    };

    let bar = Paragraph::new(help).style(theme::bar_bg());
    frame.render_widget(bar, area);
}

/// Draw the autocomplete popup above the input area (slash commands or @path).
fn draw_suggestions(frame: &mut Frame, app: &App, input_area: Rect) {
    if app.suggestions.is_empty() {
        return;
    }

    let is_path = app.suggestion_kind == SuggestionKind::Path;
    let count = app.suggestions.len() as u16;
    let height = count + 2; // +2 for top/bottom border
    let base_width: u16 = if is_path { 50 } else { 40 };
    let width = base_width.min(input_area.width);
    let y = input_area.y.saturating_sub(height);

    let popup_area = Rect::new(input_area.x, y, width, height);

    // Clear the area behind the popup
    frame.render_widget(Clear, popup_area);

    // Column width for the left side (name/path)
    let name_col: usize = if is_path { 34 } else { 12 };

    let items: Vec<Line> = app
        .suggestions
        .iter()
        .enumerate()
        .map(|(i, (cmd, desc))| {
            if i == app.suggestion_idx {
                Line::from(vec![
                    Span::styled(
                        format!(" {:<w$}", cmd, w = name_col),
                        Style::default()
                            .fg(theme::SODIUM)
                            .bg(theme::FORM_RAISED)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!(" {}", desc),
                        Style::default()
                            .fg(theme::CHALK_MID)
                            .bg(theme::FORM_RAISED),
                    ),
                ])
            } else {
                Line::from(vec![
                    Span::styled(
                        format!(" {:<w$}", cmd, w = name_col),
                        Style::default().fg(theme::CHALK_DIM),
                    ),
                    Span::styled(
                        format!(" {}", desc),
                        Style::default().fg(theme::ASH_MID),
                    ),
                ])
            }
        })
        .collect();

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::ASH));
    let paragraph = Paragraph::new(items)
        .block(block)
        .style(Style::default().bg(theme::FORM_DEEP));
    frame.render_widget(paragraph, popup_area);
}

/// Draw the session picker popup centered in the history area.
fn draw_session_picker(frame: &mut Frame, app: &App, history_area: Rect) {
    if app.session_picker.is_empty() {
        return;
    }

    let max_visible = 15u16;
    let count = app.session_picker.len() as u16;
    let visible = count.min(max_visible);
    let height = visible + 2; // +2 for borders
    let width = 80u16.min(history_area.width.saturating_sub(4));

    // Center in history area
    let x = history_area.x + (history_area.width.saturating_sub(width)) / 2;
    let y = history_area.y + (history_area.height.saturating_sub(height)) / 2;
    let popup_area = Rect::new(x, y, width, height);

    frame.render_widget(Clear, popup_area);

    // If we have more items than visible, scroll the view to keep selection visible
    let scroll_offset = if app.session_picker_idx >= visible as usize {
        app.session_picker_idx - visible as usize + 1
    } else {
        0
    };

    // Layout: " > 8h ago  First user message truncated...  5 msgs "
    let time_col = 10; // fixed width for relative time
    let count_col = 8; // fixed width for "99 msgs"
    // inner_w = width - 2 (borders) - 3 (prefix " > " or "   ") - 2 (gaps around message)
    let inner_w = (width as usize).saturating_sub(2 + 3 + time_col + count_col + 2);

    let items: Vec<Line> = app
        .session_picker
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(visible as usize)
        .map(|(i, session)| {
            let selected = i == app.session_picker_idx;
            let prefix = if selected { " > " } else { "   " };

            let time_str = session.relative_time();
            let msg_count = format!("{} msgs", session.message_count);

            // Truncate the first message to fit, collapsing newlines
            let preview: String = session
                .first_message
                .chars()
                .map(|c| if c == '\n' { ' ' } else { c })
                .take(inner_w)
                .collect();
            let preview = if session.first_message.chars().count() > inner_w {
                format!("{}\u{2026}", &preview[..preview.len().min(preview.len())])
            } else {
                preview
            };

            let (time_style, msg_style, count_style, prefix_style) = if selected {
                let bg = theme::FORM_RAISED;
                (
                    Style::default().fg(theme::SODIUM).bg(bg),
                    Style::default().fg(theme::CHALK).bg(bg),
                    Style::default().fg(theme::ASH_MID).bg(bg),
                    Style::default().fg(theme::SODIUM).bg(bg).add_modifier(Modifier::BOLD),
                )
            } else {
                (
                    Style::default().fg(theme::ASH_MID),
                    Style::default().fg(theme::CHALK_DIM),
                    Style::default().fg(theme::ASH),
                    Style::default().fg(theme::CHALK_DIM),
                )
            };

            Line::from(vec![
                Span::styled(prefix, prefix_style),
                Span::styled(format!("{:<tw$}", time_str, tw = time_col), time_style),
                Span::styled(format!("{:<mw$}", preview, mw = inner_w), msg_style),
                Span::styled(format!("{:>cw$} ", msg_count, cw = count_col), count_style),
            ])
        })
        .collect();

    let title = format!(" Sessions ({}) ", app.session_picker.len());
    let block = Block::default()
        .title(Span::styled(title, Style::default().fg(theme::SODIUM)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::ASH));
    let paragraph = Paragraph::new(items)
        .block(block)
        .style(Style::default().bg(theme::FORM_DEEP));
    frame.render_widget(paragraph, popup_area);
}

/// Draw the skill picker popup centered in the history area.
fn draw_skill_picker(frame: &mut Frame, app: &App, history_area: Rect) {
    if app.skill_picker.is_empty() {
        return;
    }

    let max_visible = 15u16;
    let count = app.skill_picker.len() as u16;
    let visible = count.min(max_visible);
    let height = visible + 2; // +2 for borders
    let width = 60u16.min(history_area.width.saturating_sub(4));

    // Center in history area
    let x = history_area.x + (history_area.width.saturating_sub(width)) / 2;
    let y = history_area.y + (history_area.height.saturating_sub(height)) / 2;
    let popup_area = Rect::new(x, y, width, height);

    frame.render_widget(Clear, popup_area);

    // Scroll to keep selection visible
    let scroll_offset = if app.skill_picker_idx >= visible as usize {
        app.skill_picker_idx - visible as usize + 1
    } else {
        0
    };

    let name_col = 20usize;
    let desc_w = (width as usize).saturating_sub(2 + 3 + name_col + 1); // borders + prefix + name + gap

    let items: Vec<Line> = app
        .skill_picker
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(visible as usize)
        .map(|(i, (name, desc))| {
            let selected = i == app.skill_picker_idx;
            let prefix = if selected { " > " } else { "   " };

            let desc_truncated: String = desc.chars().take(desc_w).collect();

            let (name_style, desc_style, prefix_style) = if selected {
                let bg = theme::FORM_RAISED;
                (
                    Style::default().fg(theme::SODIUM).bg(bg).add_modifier(Modifier::BOLD),
                    Style::default().fg(theme::CHALK).bg(bg),
                    Style::default().fg(theme::SODIUM).bg(bg).add_modifier(Modifier::BOLD),
                )
            } else {
                (
                    Style::default().fg(theme::CHALK_DIM),
                    Style::default().fg(theme::ASH_MID),
                    Style::default().fg(theme::CHALK_DIM),
                )
            };

            Line::from(vec![
                Span::styled(prefix, prefix_style),
                Span::styled(format!("{:<w$}", name, w = name_col), name_style),
                Span::styled(format!(" {}", desc_truncated), desc_style),
            ])
        })
        .collect();

    let title = format!(" Skills ({}) ", app.skill_picker.len());
    let block = Block::default()
        .title(Span::styled(title, Style::default().fg(theme::SODIUM)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::ASH));
    let paragraph = Paragraph::new(items)
        .block(block)
        .style(Style::default().bg(theme::FORM_DEEP));
    frame.render_widget(paragraph, popup_area);
}

/// Draw the agent prompt dialog (ask()) centered in the history area.
fn draw_prompt(frame: &mut Frame, app: &App, history_area: Rect) {
    let prompt = match &app.prompt {
        Some(p) => p,
        None => return,
    };

    let has_options = !prompt.options.is_empty();
    let option_count = if has_options { prompt.options.len() + 1 } else { 0 }; // +1 for "Other"

    // Count question lines (manual wrapping)
    let inner_w = 46usize; // 50 - 4 (borders + padding)
    let q_lines = if prompt.question.is_empty() {
        1
    } else {
        (prompt.question.chars().count() + inner_w - 1) / inner_w
    };

    // Calculate content height
    let extra_h: u16 = if prompt.editing_extra || !has_options { 3 } else { 0 };
    let content_h = 1 // top padding
        + q_lines as u16 // question
        + 1 // blank after question
        + option_count as u16
        + if has_options && extra_h == 0 { 1 } else { 0 } // blank after options (only if no extra area)
        + extra_h
        + 1 // blank before help
        + 1; // help bar
    let height = (content_h + 2).min(history_area.height); // +2 for borders
    let width = 50u16.min(history_area.width.saturating_sub(4));

    // Center in history area
    let x = history_area.x + (history_area.width.saturating_sub(width)) / 2;
    let y = history_area.y + (history_area.height.saturating_sub(height)) / 2;
    let popup_area = Rect::new(x, y, width, height);

    // Clear and fill background explicitly
    frame.render_widget(Clear, popup_area);

    let mut lines: Vec<Line> = Vec::new();

    // Top padding
    lines.push(Line::from(""));

    // Question text (wrap manually)
    let q_chars: Vec<char> = prompt.question.chars().collect();
    if q_chars.is_empty() {
        lines.push(Line::from(""));
    } else {
        let mut start = 0;
        while start < q_chars.len() {
            let end = (start + inner_w).min(q_chars.len());
            let chunk: String = q_chars[start..end].iter().collect();
            lines.push(Line::from(Span::styled(
                format!("  {}", chunk),
                Style::default().fg(theme::CHALK).add_modifier(Modifier::BOLD),
            )));
            start = end;
        }
    }

    lines.push(Line::from("")); // blank after question

    // Option list
    if has_options {
        for (i, opt) in prompt.options.iter().enumerate() {
            let selected = i == prompt.selected_idx;
            let marker = if selected { "\u{203a}" } else { " " };
            let label = format!("{}. {}", i + 1, opt);
            let truncated: String = label.chars().take(inner_w.saturating_sub(3)).collect();

            if selected {
                lines.push(Line::from(vec![
                    Span::styled(
                        format!(" {} ", marker),
                        Style::default().fg(theme::SODIUM).bg(theme::FORM_RAISED),
                    ),
                    Span::styled(
                        truncated,
                        Style::default().fg(theme::SODIUM).bg(theme::FORM_RAISED).add_modifier(Modifier::BOLD),
                    ),
                ]));
            } else {
                lines.push(Line::from(Span::styled(
                    format!(" {} {}", marker, truncated),
                    Style::default().fg(theme::CHALK_DIM),
                )));
            }
        }

        // "Other" option — always last
        let other_selected = prompt.selected_idx == prompt.options.len();
        let marker = if other_selected { "\u{203a}" } else { " " };
        if other_selected {
            lines.push(Line::from(vec![
                Span::styled(
                    format!(" {} ", marker),
                    Style::default().fg(theme::SODIUM).bg(theme::FORM_RAISED),
                ),
                Span::styled(
                    "Other",
                    Style::default().fg(theme::SODIUM).bg(theme::FORM_RAISED).add_modifier(Modifier::BOLD),
                ),
            ]));
        } else {
            lines.push(Line::from(Span::styled(
                format!(" {} Other", marker),
                Style::default().fg(theme::CHALK_DIM),
            )));
        }

        if extra_h == 0 {
            lines.push(Line::from("")); // blank after options
        }
    }

    // Extra text input area (shown when Shift+Tab active, or always for freeform)
    if prompt.editing_extra || !has_options {
        let label = if has_options { " Extra context " } else { " Your answer " };
        let rule_w = inner_w.saturating_sub(label.chars().count() + 3);
        lines.push(Line::from(Span::styled(
            format!("  \u{250c}{}\u{2500}{}", label, "\u{2500}".repeat(rule_w)),
            Style::default().fg(theme::ASH_MID),
        )));
        let text_display: String = prompt.extra_text.chars().take(inner_w.saturating_sub(4)).collect();
        lines.push(Line::from(vec![
            Span::styled("  \u{2502} ", Style::default().fg(theme::ASH_MID)),
            Span::styled(text_display, Style::default().fg(theme::CHALK)),
            Span::styled("\u{2588}", Style::default().fg(theme::SODIUM)),
        ]));
        lines.push(Line::from(Span::styled(
            format!("  \u{2514}{}", "\u{2500}".repeat(inner_w.saturating_sub(2))),
            Style::default().fg(theme::ASH_MID),
        )));
    }

    lines.push(Line::from("")); // blank before help

    // Help bar
    let help = if has_options {
        Line::from(vec![
            Span::styled("  \u{2191}\u{2193}", theme::help_key()),
            Span::styled(" select  ", theme::help_desc()),
            Span::styled("\u{23ce}", theme::help_key()),
            Span::styled(" submit  ", theme::help_desc()),
            Span::styled("\u{21e7}\u{21e5}", theme::help_key()),
            Span::styled(" context  ", theme::help_desc()),
            Span::styled("Esc", theme::help_key()),
            Span::styled(" cancel", theme::help_desc()),
        ])
    } else {
        Line::from(vec![
            Span::styled("  \u{23ce}", theme::help_key()),
            Span::styled(" submit  ", theme::help_desc()),
            Span::styled("Esc", theme::help_key()),
            Span::styled(" cancel", theme::help_desc()),
        ])
    };
    lines.push(help);

    let title = " Agent Question ";
    let block = Block::default()
        .title(Span::styled(title, Style::default().fg(theme::SODIUM).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::ASH_LIGHT));
    let paragraph = Paragraph::new(lines)
        .block(block)
        .style(Style::default().bg(theme::FORM_DEEP));
    frame.render_widget(paragraph, popup_area);
}

/// Count total visual lines the input text occupies, accounting for wrapping.
/// Each logical line (split by '\n') wraps at `width` columns.
fn input_visual_lines(input: &str, width: usize) -> usize {
    if width == 0 {
        return 1;
    }
    let mut total = 0;
    for line in input.split('\n') {
        let len = line.chars().count();
        if len == 0 {
            total += 1;
        } else {
            total += (len + width - 1) / width;
        }
    }
    total.max(1)
}

/// Compute the visual (row, col) of the cursor in the input, accounting for wrapping.
/// `full_width` is the total rendering width of the paragraph inner area.
/// The returned column is absolute (includes the 2-char prefix on the first visual line
/// of each logical line, and starts from 0 on wrapped continuation lines).
fn input_cursor_position(input: &str, cursor_pos: usize, full_width: usize) -> (usize, usize) {
    let prefix = 2usize; // "❯ " or "  " prefix on each logical line
    let before_cursor = &input[..cursor_pos];
    let mut vis_row: usize = 0;

    // Process each logical line before the cursor's line
    let last_newline = before_cursor.rfind('\n').map(|i| i + 1).unwrap_or(0);
    if last_newline > 0 {
        for line in input[..last_newline - 1].split('\n') {
            let total = UnicodeWidthStr::width(line) + prefix;
            if full_width > 0 {
                vis_row += (total.max(1) + full_width - 1) / full_width;
            } else {
                vis_row += 1;
            }
        }
    }

    // Column position within the current logical line (visual width, not char count)
    let col = UnicodeWidthStr::width(&input[last_newline..cursor_pos]);
    // Absolute position including prefix — matches how ratatui wraps the Line content
    let abs_pos = prefix + col;

    if full_width > 0 && abs_pos > 0 {
        vis_row += abs_pos / full_width;
        (vis_row, abs_pos % full_width)
    } else {
        (vis_row, abs_pos)
    }
}
