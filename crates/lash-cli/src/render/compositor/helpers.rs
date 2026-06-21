fn draw_lines_region(frame: &mut Frame<'_>, area: Rect, lines: &[Line<'static>], style: Style) {
    frame.fill(area, ' ', style);
    for (idx, line) in lines.iter().enumerate().take(area.height as usize) {
        frame.write_line(area.x, area.y + idx as u16, line, area.width);
    }
}

fn draw_top_bottom_rule(frame: &mut Frame<'_>, area: Rect, style: Style) {
    for x in 0..area.width {
        frame.write_text(area.x + x, area.y, "─", style, 1);
        frame.write_text(
            area.x + x,
            area.y + area.height.saturating_sub(1),
            "─",
            style,
            1,
        );
    }
}

fn fg(color: lash_tui::Color) -> Style {
    Style::default().fg(color)
}
