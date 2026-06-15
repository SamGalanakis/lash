#[derive(Clone)]
struct StatusSlot {
    spans: Vec<Span<'static>>,
    /// Higher = more important. Dropped last under width pressure.
    priority: u8,
}

impl StatusSlot {
    fn width(&self) -> usize {
        self.spans
            .iter()
            .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
            .sum()
    }
}

/// Greedy knapsack by descending priority. Returns the slots that fit,
/// preserving their original order.
fn fit_slots(slots: &[StatusSlot], budget: usize) -> Vec<Span<'static>> {
    if budget == 0 || slots.is_empty() {
        return Vec::new();
    }
    if slots.len() > u64::BITS as usize {
        return fit_slots_fallback(slots, budget);
    }

    let mut included = 0u64;
    let mut visited = 0u64;
    let mut used = 0usize;
    while (visited.count_ones() as usize) < slots.len() {
        let mut best_idx = None;
        let mut best_priority = 0u8;
        for (idx, slot) in slots.iter().enumerate() {
            let bit = 1u64 << idx;
            if visited & bit != 0 {
                continue;
            }
            if best_idx.is_none() || slot.priority > best_priority {
                best_idx = Some(idx);
                best_priority = slot.priority;
            }
        }
        let Some(idx) = best_idx else {
            break;
        };
        let bit = 1u64 << idx;
        visited |= bit;
        let width = slots[idx].width();
        if width > 0 && used + width <= budget {
            included |= bit;
            used += width;
        }
    }

    let span_count = slots
        .iter()
        .enumerate()
        .filter(|(idx, _)| included & (1u64 << idx) != 0)
        .map(|(_, slot)| slot.spans.len())
        .sum();
    let mut spans = Vec::with_capacity(span_count);
    for (idx, slot) in slots.iter().enumerate() {
        if included & (1u64 << idx) != 0 {
            spans.extend(slot.spans.iter().cloned());
        }
    }
    spans
}

fn fit_slots_fallback(slots: &[StatusSlot], budget: usize) -> Vec<Span<'static>> {
    let mut indices: Vec<usize> = (0..slots.len()).collect();
    indices.sort_by(|a, b| slots[*b].priority.cmp(&slots[*a].priority));

    let mut included = vec![false; slots.len()];
    let mut used = 0usize;
    for idx in indices {
        let width = slots[idx].width();
        if width == 0 {
            continue;
        }
        if used + width <= budget {
            included[idx] = true;
            used += width;
        }
    }

    let span_count = slots
        .iter()
        .enumerate()
        .filter(|(idx, _)| included[*idx])
        .map(|(_, slot)| slot.spans.len())
        .sum();
    let mut spans = Vec::with_capacity(span_count);
    for (idx, slot) in slots.iter().enumerate() {
        if included[idx] {
            spans.extend(slot.spans.iter().cloned());
        }
    }
    spans
}

fn draw_status_bar(frame: &mut Frame<'_>, app: &App, area: Rect) {
    frame.fill(area, ' ', bg(theme::surface_raised()));
    if area.width == 0 {
        return;
    }

    let (left, right) = build_status_slots(app);
    let total = area.width as usize;

    // Right gets first refusal with up to half the bar. Whatever it leaves
    // (minus one column for visual breathing room) becomes the left budget.
    let right_spans = fit_slots(&right, total / 2);
    let right_width: usize = right_spans
        .iter()
        .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
        .sum();
    let gap = if right_width > 0 { 1 } else { 0 };
    let left_budget = total.saturating_sub(right_width + gap);
    let left_spans = fit_slots(&left, left_budget);

    if !left_spans.is_empty() {
        let line = Line::from(left_spans);
        let width = line.width() as u16;
        frame.write_line(0, 0, &line, width);
    }
    if !right_spans.is_empty() {
        let line = Line::from(right_spans);
        let width = line.width() as u16;
        let x = area.width.saturating_sub(width);
        frame.write_line(x, 0, &line, width);
    }
}

fn build_status_slots(app: &App) -> (Vec<StatusSlot>, Vec<StatusSlot>) {
    let sep_style = theme::text_faint_style();

    // LEFT side: brand · model · execution mode · variant
    let mut left = Vec::new();
    left.push(StatusSlot {
        spans: vec![
            Span::raw(" "),
            Span::styled(
                "lash",
                Style::default()
                    .fg(theme::brand())
                    .add_modifier(Modifier::Bold),
            ),
        ],
        priority: 100, // always keep — identity anchor
    });
    if !app.model.is_empty() {
        left.push(StatusSlot {
            spans: vec![
                Span::styled(" · ", sep_style),
                Span::styled(app.model.clone(), theme::text_subtle_style()),
            ],
            priority: 80,
        });
    }
    if !app.execution_mode_label.is_empty() {
        left.push(StatusSlot {
            spans: vec![
                Span::styled(" · ", sep_style),
                Span::styled(app.execution_mode_label.clone(), theme::text_subtle_style()),
            ],
            priority: 70,
        });
    }
    if let Some(variant) = app
        .model_variant
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        left.push(StatusSlot {
            spans: vec![
                Span::styled(" · ", sep_style),
                Span::styled(
                    variant.to_string(),
                    Style::default()
                        .fg(theme::brand())
                        .add_modifier(Modifier::Bold),
                ),
            ],
            priority: 40, // drop before model under pressure
        });
    }

    // RIGHT side: context-window meter. The expand keybind is searchable
    // via `/controls` and the `▸` marker on tool activities — no teaching
    // slot needed in the status bar.
    let mut right = Vec::new();
    if let Some(ctx) = status_bar_context_spans(app) {
        right.push(ctx);
    }

    (left, right)
}

fn status_bar_context_spans(app: &App) -> Option<StatusSlot> {
    // Don't add `live_output_tokens_estimate` to either branch's total: on the
    // very first turn, before `last_response_usage` lands, that's the only
    // nonzero number — and using it as the displayed total reads as if the
    // streamed output bytes were the entire context, producing nonsense like
    // `36 · 0%` against a 1.1M-token window. Wait for real input accounting.
    let Some(context_window) = app.usage.context_window else {
        let total = app.usage.token_usage.input_tokens + app.usage.token_usage.output_tokens;
        if total <= 0 {
            return None;
        }
        return Some(StatusSlot {
            spans: vec![
                Span::styled("ctx ", theme::text_faint_style()),
                Span::styled(format_tokens(total), theme::text_subtle_style()),
                Span::raw(" "),
            ],
            priority: 70,
        });
    };

    let used = current_context_budget_tokens(app)
        .or_else(|| {
            app.usage
                .last_prompt_usage
                .as_ref()
                .map(|usage| usage.context_budget_tokens as i64)
                .filter(|used| *used > 0)
        })
        .or_else(|| {
            let total = app.usage.token_usage.input_tokens + app.usage.token_usage.output_tokens;
            (total > 0).then_some(total)
        })?;

    let pct = if context_window == 0 {
        0.0
    } else {
        used as f64 / context_window as f64 * 100.0
    };

    // The old layout was `3.4k / 1.1M (9.3%)` — three representations of the
    // same number: tokens used, total window, and the derived percentage.
    // The percentage is the only thing you can act on (it tells you how close
    // you are to the wall); the raw token count is useful for debugging.
    // Keep just those two, separated by the faint middle-dot used elsewhere
    // in the bar. Integer percentage — `9.3%` vs `9%` is false precision at
    // this scale.
    Some(StatusSlot {
        spans: vec![
            Span::styled("ctx ", theme::text_faint_style()),
            Span::styled(format_tokens(used), theme::text_subtle_style()),
            Span::styled(" · ", theme::text_faint_style()),
            Span::styled(format!("{pct:.0}%"), theme::text_subtle_style()),
            Span::raw(" "),
        ],
        priority: 70,
    })
}
