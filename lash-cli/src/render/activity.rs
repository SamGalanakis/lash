use super::artifact::render_snippet_preview_with_indent;
use super::*;

fn activity_style(status: ActivityStatus) -> Style {
    match status {
        ActivityStatus::Completed => theme::tool_success(),
        ActivityStatus::Failed => theme::tool_failure(),
        ActivityStatus::Running => theme::turn_status_state(),
        ActivityStatus::Partial => theme::text_subtle_style(),
    }
}

/// True if the activity has content that the current `expand_level` is
/// NOT rendering. Drives the `▸` indicator on the call line so the
/// reader can tell when they're looking at a compact view.
///
/// Level mapping (see `docs/design-language.html`):
/// * L0 shows summary + detail lines + QuestionPanel + compact patch
///   (filename + `+/-` counts, no diff body) + snippet preview.
/// * L1 adds full patch diffs (capped by file count).
/// * L2 adds DiffPreview / TextPreview / SourceList / reasoning /
///   shell output / full patch diffs without a file cap.
fn has_hidden_content_at(activity: &ActivityBlock, expand_level: u8) -> bool {
    if expand_level >= 2 {
        return false;
    }
    match activity.result.artifact.as_ref() {
        Some(ActivityArtifact::PatchPreview { files, .. }) => {
            let non_empty_diffs = files.iter().any(|file| !file.diff.trim().is_empty());
            let over_cap = files.len() > COMPACT_PATCH_PREVIEW_MAX_FILES;
            if expand_level >= 1 {
                // L1 renders diffs but caps files; hidden only when
                // there are more files than the cap.
                over_cap
            } else {
                // L0 drops diffs entirely; hidden when either diffs
                // exist or the file list is truncated.
                non_empty_diffs || over_cap
            }
        }
        Some(
            ActivityArtifact::DiffPreview { .. }
            | ActivityArtifact::TextPreview { .. }
            | ActivityArtifact::SourceList { .. },
        ) => true,
        Some(ActivityArtifact::SnippetPreview(_) | ActivityArtifact::QuestionPanel(_)) | None => {
            false
        }
    }
}

fn activity_prefix(activity: &ActivityBlock) -> (&'static str, Style, Style) {
    // Weight note: generic activity rows use the bullet (`•`) rather
    // than the middle-dot (`·`) so a completed tool call pops against
    // reasoning's `┊` gutter and the dock's `▶`/`■`/`□` glyphs. The
    // middle dot is too visually light to read as "a thing happened".
    match activity.call.kind {
        ActivityKind::Exploration => ("• ", theme::explore_marker(), theme::explore_label()),
        ActivityKind::Edit => (
            "• ",
            theme::edit_lane_bold(),
            theme::assistant_text().add_modifier(Modifier::Bold),
        ),
        ActivityKind::Ask => (
            "◆ ",
            theme::subagent_marker(),
            Style::default()
                .fg(theme::brand())
                .add_modifier(Modifier::Bold),
        ),
        ActivityKind::Subagent => (
            "◆ ",
            theme::subagent_marker(),
            Style::default().fg(theme::text_muted()),
        ),
        _ => match activity.result.status {
            ActivityStatus::Completed => (
                "• ",
                theme::tool_success(),
                activity_style(activity.result.status),
            ),
            ActivityStatus::Failed => (
                "× ",
                theme::tool_failure(),
                activity_style(activity.result.status),
            ),
            ActivityStatus::Running => (
                "◆ ",
                theme::turn_status_slash(),
                activity_style(activity.result.status),
            ),
            ActivityStatus::Partial => (
                "◐ ",
                theme::text_subtle_style(),
                activity_style(activity.result.status),
            ),
        },
    }
}

pub(super) fn render_activity_block(
    activity: &ActivityBlock,
    expand_level: u8,
    lines: &mut Vec<Line<'static>>,
    viewport_width: usize,
) {
    if activity.call.kind == ActivityKind::Hidden {
        if expand_level >= 2
            && let Some(artifact) = &activity.result.artifact
        {
            render_activity_artifact(lines, artifact, viewport_width, "    ");
        }
        return;
    }

    // QuestionPanel takes over the activity row with its own
    // interactive panel and is always shown — it's the ask dialog
    // the user already answered, not something to fold away.
    if let Some(ActivityArtifact::QuestionPanel(panel)) = activity.result.artifact.as_ref() {
        render_question_panel_artifact(panel, lines, viewport_width);
        return;
    }

    let (prefix, prefix_style, summary_style) = activity_prefix(activity);
    let prefix_width = UnicodeWidthStr::width(prefix);
    let duration_text = crate::util::format_duration_ms_if_visible(activity.duration_ms);
    let hidden_content = has_hidden_content_at(activity, expand_level);
    // Call line is composed from up to six independently styled regions:
    //   [sigil] [tag] · [summary body] · [duration] [▸]
    // The tag ("EXPLORE", etc.) used to be glued into the summary via a
    // flat ` · ` separator. It's now a structured `ActivityCall::tag`
    // field so the renderer can style it in brand weight while the body
    // stays in subtle text, and the ` · ` separators are rendered as
    // faint spans instead of being characters in a string.
    let tag_text = activity.call.tag.as_deref();
    let tag_width = tag_text.map(UnicodeWidthStr::width).unwrap_or(0);
    let tag_separator_width = if tag_text.is_some() { 3 } else { 0 };
    let duration_width = duration_text
        .as_deref()
        .map(UnicodeWidthStr::width)
        .unwrap_or(0);
    let duration_separator_width = if duration_text.is_some() { 3 } else { 0 };
    let hidden_marker_width = if hidden_content { 2 } else { 0 };
    let available = viewport_width
        .saturating_sub(prefix_width)
        .saturating_sub(tag_width + tag_separator_width)
        .saturating_sub(duration_width + duration_separator_width)
        .saturating_sub(hidden_marker_width);
    let mut spans = vec![Span::styled(prefix.to_string(), prefix_style)];
    if let Some(tag) = tag_text {
        spans.push(Span::styled(
            tag.to_string(),
            Style::default()
                .fg(theme::brand())
                .add_modifier(Modifier::Bold),
        ));
        spans.push(Span::styled(" · ", theme::text_faint_style()));
    }
    spans.push(Span::styled(
        truncate_to_display_width(&activity.call.summary, available),
        summary_style,
    ));
    if let Some(duration_text) = duration_text {
        spans.push(Span::styled(" · ", theme::text_faint_style()));
        spans.push(Span::styled(duration_text, theme::text_subtle_style()));
    }
    if hidden_content {
        // Subtle right-pointing triangle tells the reader there's more
        // content folded under this activity — details, diff, text preview,
        // or source list that the current expand level isn't showing.
        spans.push(Span::styled(" ▸", theme::text_faint_style()));
    }
    lines.push(Line::from(spans));

    let detail_prefix = "    ";
    let detail_prefix_width = UnicodeWidthStr::width(detail_prefix);

    // ── L0+: detail lines, parallel/subagent children, compact
    //        patch (no diffs) + snippet preview.
    if activity.call.kind == ActivityKind::Exploration {
        render_recent_activity_feed_lines(
            lines,
            &activity.result.detail_lines,
            viewport_width,
            detail_prefix,
            theme::explore_marker(),
            "exploration step",
        );
    } else {
        // Wrap-continuation gets two extra columns so a long detail
        // that spills onto the next row can't be confused with a
        // fresh detail item starting at the prefix column.
        let continuation_prefix = format!("{detail_prefix}  ");
        for detail in &activity.result.detail_lines {
            push_wrapped_prefixed(
                lines,
                detail_prefix.to_string(),
                continuation_prefix.clone(),
                detail,
                Style::default().fg(theme::text_faint()),
                viewport_width.saturating_sub(detail_prefix_width) + detail_prefix_width,
            );
        }
    }

    if activity.call.kind == ActivityKind::Parallel {
        for child in &activity.children {
            push_wrapped_prefixed(
                lines,
                "      ".to_string(),
                "      ".to_string(),
                &child.call.summary,
                activity_style(child.result.status),
                viewport_width,
            );
        }
    }

    if activity.call.kind == ActivityKind::Subagent {
        for child in &activity.children {
            push_wrapped_prefixed(
                lines,
                detail_prefix.to_string(),
                detail_prefix.to_string(),
                &child.call.summary,
                match child.result.status {
                    ActivityStatus::Failed => theme::error(),
                    ActivityStatus::Running => theme::turn_status_state(),
                    ActivityStatus::Partial => theme::text_subtle_style(),
                    ActivityStatus::Completed => theme::subagent_child(),
                },
                viewport_width,
            );
        }
    }

    if let Some(artifact) = &activity.result.artifact {
        match artifact {
            ActivityArtifact::PatchPreview { files, .. } => {
                // L0: file summaries only (no diff body). L1: add
                // inline diffs, capped by file count. L2: no cap.
                let include_diffs = expand_level >= 1;
                let max_files = if expand_level >= 2 {
                    None
                } else {
                    Some(COMPACT_PATCH_PREVIEW_MAX_FILES)
                };
                render_patch_artifact(
                    lines,
                    files,
                    viewport_width,
                    detail_prefix,
                    include_diffs,
                    max_files,
                );
            }
            ActivityArtifact::SnippetPreview(preview) => {
                render_snippet_preview_with_indent(preview, lines, viewport_width, detail_prefix);
            }
            _ => {}
        }
    }

    // ── L2+: heavy artifact bodies (inline diffs, captured text,
    //        source lists). Patch/snippet are already rendered above
    //        at their level-appropriate form.
    if expand_level >= 2
        && let Some(
            artifact @ (ActivityArtifact::DiffPreview { .. }
            | ActivityArtifact::TextPreview { .. }
            | ActivityArtifact::SourceList { .. }),
        ) = activity.result.artifact.as_ref()
    {
        render_activity_artifact(lines, artifact, viewport_width, detail_prefix);
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
    let detail_style = Style::default().fg(theme::text_faint());
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
            render_snippet_preview_with_indent(preview, lines, viewport_width, indent);
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
                Style::default().fg(theme::text_faint()),
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
