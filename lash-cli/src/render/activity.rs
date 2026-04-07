use super::*;

fn activity_style(status: ActivityStatus) -> Style {
    match status {
        ActivityStatus::Completed => theme::tool_success(),
        ActivityStatus::Failed => theme::tool_failure(),
    }
}

fn activity_prefix(activity: &ActivityBlock) -> (&'static str, Style, Style) {
    match activity.kind {
        ActivityKind::Exploration => ("· ", theme::explore_marker(), theme::explore_label()),
        ActivityKind::Edit => (
            "· ",
            theme::edit_lane_bold(),
            theme::assistant_text().add_modifier(Modifier::Bold),
        ),
        ActivityKind::Ask => (
            "◆ ",
            theme::delegate_marker(),
            Style::default()
                .fg(theme::SODIUM)
                .add_modifier(Modifier::Bold),
        ),
        ActivityKind::Delegate => (
            "◆ ",
            theme::delegate_marker(),
            Style::default().fg(theme::CHALK_MID),
        ),
        _ => match activity.status {
            ActivityStatus::Completed => {
                ("· ", theme::tool_success(), activity_style(activity.status))
            }
            ActivityStatus::Failed => {
                ("× ", theme::tool_failure(), activity_style(activity.status))
            }
        },
    }
}

pub(super) fn render_activity_block(
    activity: &ActivityBlock,
    expand_level: u8,
    lines: &mut Vec<Line<'static>>,
    viewport_width: usize,
) {
    if expand_level >= 1
        && let Some(ActivityArtifact::QuestionPanel(panel)) = activity.artifact.as_ref()
    {
        render_question_panel_artifact(panel, lines, viewport_width);
        return;
    }

    let (prefix, prefix_style, summary_style) = activity_prefix(activity);
    let prefix_width = UnicodeWidthStr::width(prefix);
    let summary = if let Some(duration_text) =
        crate::util::format_duration_ms_if_visible(activity.duration_ms)
    {
        format!("{} · {}", activity.summary, duration_text)
    } else {
        activity.summary.clone()
    };
    lines.push(Line::from(vec![
        Span::styled(prefix.to_string(), prefix_style),
        Span::styled(
            truncate_to_display_width(&summary, viewport_width.saturating_sub(prefix_width)),
            summary_style,
        ),
    ]));

    let detail_prefix = "    ";
    let detail_prefix_width = UnicodeWidthStr::width(detail_prefix);

    if expand_level >= 1 {
        if activity.kind == ActivityKind::Exploration {
            render_recent_activity_feed_lines(
                lines,
                &activity.detail_lines,
                viewport_width,
                detail_prefix,
                theme::explore_marker(),
                "exploration step",
            );
        } else {
            for detail in &activity.detail_lines {
                push_wrapped_prefixed(
                    lines,
                    detail_prefix.to_string(),
                    detail_prefix.to_string(),
                    detail,
                    Style::default().fg(theme::ASH_TEXT),
                    viewport_width.saturating_sub(detail_prefix_width) + detail_prefix_width,
                );
            }
        }

        if activity.kind == ActivityKind::Parallel {
            for child in &activity.children {
                push_wrapped_prefixed(
                    lines,
                    "      ".to_string(),
                    "      ".to_string(),
                    &child.summary,
                    activity_style(child.status),
                    viewport_width,
                );
            }
        }

        if activity.kind == ActivityKind::Delegate {
            for child in &activity.children {
                push_wrapped_prefixed(
                    lines,
                    detail_prefix.to_string(),
                    detail_prefix.to_string(),
                    &child.summary,
                    if child.status == ActivityStatus::Failed {
                        theme::error()
                    } else {
                        theme::delegate_child()
                    },
                    viewport_width,
                );
            }
        }

        if expand_level == 1
            && let Some(artifact) = &activity.artifact
        {
            match artifact {
                ActivityArtifact::PatchPreview { files, .. } => {
                    render_patch_artifact(
                        lines,
                        files,
                        viewport_width,
                        detail_prefix,
                        true,
                        Some(COMPACT_PATCH_PREVIEW_MAX_FILES),
                    );
                }
                ActivityArtifact::SnippetPreview(preview) => {
                    render_snippet_preview(preview, lines, viewport_width);
                }
                _ => {}
            }
        }
    }

    if expand_level >= 2
        && let Some(artifact) = &activity.artifact
    {
        match artifact {
            ActivityArtifact::PatchPreview { files, .. } => {
                render_patch_artifact(lines, files, viewport_width, detail_prefix, true, None);
            }
            _ => render_activity_artifact(lines, artifact, viewport_width, detail_prefix),
        }
    }
}

fn render_recent_activity_feed_lines(
    lines: &mut Vec<Line<'static>>,
    detail_lines: &[String],
    viewport_width: usize,
    prefix: &str,
    prefix_style: Style,
    overflow_label: &str,
) {
    let prefix_width = UnicodeWidthStr::width(prefix);
    let available = viewport_width.saturating_sub(prefix_width);
    let detail_style = Style::default().fg(theme::ASH_TEXT);
    let hidden_count = detail_lines
        .len()
        .saturating_sub(COMPACT_ACTIVITY_FEED_MAX_ITEMS);
    let visible = &detail_lines[hidden_count..];

    if hidden_count > 0 {
        lines.push(Line::from(vec![
            Span::styled(prefix.to_string(), prefix_style),
            Span::styled(
                format!(
                    "… {hidden_count} earlier {overflow_label}{} hidden …",
                    if hidden_count == 1 { "" } else { "s" }
                ),
                detail_style,
            ),
        ]));
    }

    for detail in visible {
        let segments = if detail.is_empty() {
            vec![(0usize, 0usize)]
        } else {
            text_layout::wrap_text_ranges_wordwise(detail, available)
        };
        for (segment_idx, &(start, end)) in segments
            .iter()
            .take(COMPACT_ACTIVITY_FEED_MAX_ROWS_PER_ITEM)
            .enumerate()
        {
            let is_last_visible = segment_idx + 1 == COMPACT_ACTIVITY_FEED_MAX_ROWS_PER_ITEM
                && segment_idx + 1 < segments.len();
            let chunk = if detail.is_empty() {
                String::new()
            } else if is_last_visible {
                truncate_with_forced_ellipsis(&detail[start..end], available)
            } else {
                truncate_to_display_width(&detail[start..end], available)
            };
            lines.push(Line::from(vec![
                Span::styled(prefix.to_string(), prefix_style),
                Span::styled(chunk, detail_style),
            ]));
        }
    }
}

fn render_activity_artifact(
    lines: &mut Vec<Line<'static>>,
    artifact: &ActivityArtifact,
    viewport_width: usize,
    indent: &str,
) {
    match artifact {
        ActivityArtifact::QuestionPanel(_) => {}
        ActivityArtifact::DiffPreview { title, diff } => {
            lines.push(Line::from(vec![
                Span::styled(indent.to_string(), theme::code_chrome()),
                Span::styled(format!("{title}:"), theme::code_header()),
            ]));
            render_inline_diff(lines, diff, viewport_width, &format!("{indent}  "));
        }
        ActivityArtifact::PatchPreview { files, .. } => {
            render_patch_preview(lines, files, viewport_width, indent, true, None);
        }
        ActivityArtifact::TextPreview { title, text } => {
            if let Some(title) = title {
                lines.push(Line::from(vec![
                    Span::styled(indent.to_string(), theme::code_chrome()),
                    Span::styled(format!("{title}:"), theme::code_header()),
                ]));
            }
            let prefix = format!("{indent}  ");
            for line in preview_text_lines(text) {
                push_wrapped_prefixed(
                    lines,
                    prefix.clone(),
                    prefix.clone(),
                    &line,
                    theme::system_output(),
                    viewport_width,
                );
            }
        }
        ActivityArtifact::SourceList { title, items } => {
            lines.push(Line::from(vec![
                Span::styled(indent.to_string(), theme::code_chrome()),
                Span::styled(format!("{title}:"), theme::code_header()),
            ]));
            let prefix = format!("{indent}  ");
            for item in items {
                push_wrapped_prefixed(
                    lines,
                    prefix.clone(),
                    prefix.clone(),
                    item,
                    theme::system_output(),
                    viewport_width,
                );
            }
        }
        ActivityArtifact::SnippetPreview(preview) => {
            render_snippet_preview(preview, lines, viewport_width);
        }
    }
}

fn render_patch_preview(
    lines: &mut Vec<Line<'static>>,
    files: &[PatchFilePreview],
    viewport_width: usize,
    indent: &str,
    include_diffs: bool,
    max_files: Option<usize>,
) {
    let hidden_count = max_files
        .map(|limit| files.len().saturating_sub(limit))
        .unwrap_or(0);
    if hidden_count > 0 {
        lines.push(Line::from(vec![
            Span::styled(format!("{indent}  "), theme::patch_frame()),
            Span::styled(
                format!(
                    "… {hidden_count} earlier file change{} hidden …",
                    if hidden_count == 1 { "" } else { "s" }
                ),
                Style::default().fg(theme::ASH_TEXT),
            ),
        ]));
    }

    for file in files.iter().skip(hidden_count) {
        render_patch_summary_line(lines, &format!("{indent}  "), file, viewport_width);
        if include_diffs && !file.diff.trim().is_empty() {
            render_inline_diff(lines, &file.diff, viewport_width, &format!("{indent}│ "));
        }
    }
}

fn render_patch_summary_line(
    lines: &mut Vec<Line<'static>>,
    prefix: &str,
    file: &PatchFilePreview,
    viewport_width: usize,
) {
    let subject = patch_file_subject(file);
    let label = patch_status_title(&file.status);
    let counts = format!(" (+{} -{})", file.added, file.removed);
    let available = viewport_width
        .saturating_sub(UnicodeWidthStr::width(prefix))
        .saturating_sub(UnicodeWidthStr::width(label) + 1)
        .saturating_sub(UnicodeWidthStr::width(counts.as_str()));
    lines.push(Line::from(vec![
        Span::styled(prefix.to_string(), theme::patch_frame()),
        Span::styled(label.to_string(), theme::patch_label()),
        Span::raw(" "),
        Span::styled(
            truncate_to_display_width(&subject, available),
            theme::assistant_text(),
        ),
        Span::styled(" (".to_string(), theme::code_chrome()),
        Span::styled(format!("+{}", file.added), theme::patch_add()),
        Span::raw(" "),
        Span::styled(format!("-{}", file.removed), theme::patch_remove()),
        Span::styled(")".to_string(), theme::code_chrome()),
    ]));
}

fn render_patch_artifact(
    lines: &mut Vec<Line<'static>>,
    files: &[PatchFilePreview],
    viewport_width: usize,
    indent: &str,
    include_diffs: bool,
    max_files: Option<usize>,
) {
    if files.len() == 1 {
        if include_diffs && !files[0].diff.trim().is_empty() {
            render_inline_diff(
                lines,
                &files[0].diff,
                viewport_width,
                &format!("{indent}│ "),
            );
        }
        return;
    }
    render_patch_preview(
        lines,
        files,
        viewport_width,
        indent,
        include_diffs,
        max_files,
    );
}
