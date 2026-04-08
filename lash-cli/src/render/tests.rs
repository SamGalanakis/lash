use super::artifact::render_snippet_preview;
use super::*;
use crate::activity::ActivityState;
use crate::app::projected_blocks_from_state;
use crate::theme;
use lash::ToolProvider;
use lash::tools::ShowcaseSnippet;
use lash::{Part, PartKind, PromptRequest};
use serde_json::Value;
use std::sync::mpsc;

#[test]
fn exploration_detail_lines_are_indented_under_summary() {
    let activity = ActivityBlock {
        kind: ActivityKind::Exploration,
        status: ActivityStatus::Completed,
        tool_name: "search".into(),
        summary: "EXPLORE · 1 step".into(),
        detail_lines: vec!["Read README.md".into()],
        duration_ms: 0,
        args: Value::Null,
        result: Value::Null,
        artifact: None,
        children: Vec::new(),
        extra: None,
    };
    let blocks = vec![DisplayBlock::Activity(Box::new(activity))];
    let rendered = render_block(&blocks, 0, 1, 80, 20)
        .into_iter()
        .map(|line| {
            line.spans
                .into_iter()
                .map(|span| span.content.into_owned())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    assert_eq!(rendered[0], "· EXPLORE · 1 step");
    assert_eq!(rendered[1], "    Read README.md");
}

#[test]
fn extract_history_selection_text_reads_across_scrolled_content_rows() {
    let mut app = App::new("gpt-5.4".into(), "test".into());
    app.blocks = vec![DisplayBlock::UserInput("alpha\nbeta\ngamma".into())];
    app.scroll_offset = 1;
    app.selection.anchor = (2, 1);
    app.selection.end = (4, 2);
    app.selection.visible = true;

    let selected = extract_history_selection_text(&app, 40, 9).expect("selected text");
    assert_eq!(selected, "beta\n  ga");
}

#[test]
fn prompt_question_wraps_long_path_cleanly() {
    let (response_tx, _response_rx) = mpsc::channel();
    let prompt = PromptState {
        request: PromptRequest::single(
            "Plan .lash/plans/15d5a2bd-841d-4729-8968-ae7874385e16.md is ready. Exit plan mode now?",
            vec!["Exit plan mode".into(), "Keep planning".into()],
        )
        .with_optional_note(),
        focus: crate::overlay::PromptFocus::Options,
        cursor: 0,
        scroll_offset: 0,
        selected: Default::default(),
        reply_text: String::new(),
        reply_cursor: 0,
        response_tx,
    };

    let rendered = prompt_content_lines_snapshot(&prompt, 38)
        .into_iter()
        .map(|line| {
            line.spans
                .into_iter()
                .map(|span| span.content.into_owned())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    assert_eq!(rendered[0], "Plan .lash/plans/15d5a2bd-841d-4729-89");
    assert_eq!(rendered[1], "68-ae7874385e16.md is ready. Exit plan");
    assert_eq!(rendered[2], " mode now?");
    assert_eq!(rendered[3], "");
    assert!(rendered.iter().any(|line| line.contains("Choices")));
}

#[test]
fn prompt_panel_renders_before_question_and_choices() {
    let (response_tx, _response_rx) = mpsc::channel();
    let prompt = PromptState {
        request: PromptRequest::single(
            "Exit plan mode?",
            vec!["Exit plan mode".into(), "Keep planning".into()],
        )
        .with_optional_note()
        .with_markdown_panel("PLAN", "# Plan\n\n## Steps\n- First\n- Second"),
        focus: crate::overlay::PromptFocus::Options,
        cursor: 0,
        scroll_offset: 0,
        selected: Default::default(),
        reply_text: String::new(),
        reply_cursor: 0,
        response_tx,
    };

    let rendered = prompt_content_lines_snapshot(&prompt, 42)
        .into_iter()
        .map(|line| {
            line.spans
                .into_iter()
                .map(|span| span.content.into_owned())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    let plan_idx = rendered
        .iter()
        .position(|line| line.contains("PLAN"))
        .expect("plan panel");
    let question_idx = rendered
        .iter()
        .position(|line| line.contains("Exit plan mode?"))
        .expect("question");
    let choices_idx = rendered
        .iter()
        .position(|line| line.contains("Choices"))
        .expect("choices");

    assert!(plan_idx < question_idx);
    assert!(question_idx < choices_idx);
    assert!(rendered.iter().any(|line| line.contains("First")));
    assert!(rendered.iter().any(|line| line.contains("Second")));
    assert!(!rendered.iter().any(|line| line.contains("┌")));
}

#[test]
fn prompt_panel_strips_redundant_h1_matching_panel_title() {
    let (response_tx, _response_rx) = mpsc::channel();
    let prompt = PromptState {
        request: PromptRequest::single("Exit plan mode?", vec!["Exit".into()])
            .with_markdown_panel("PLAN", "# Plan\n\n## Steps\n- First"),
        focus: crate::overlay::PromptFocus::Options,
        cursor: 0,
        scroll_offset: 0,
        selected: Default::default(),
        reply_text: String::new(),
        reply_cursor: 0,
        response_tx,
    };

    let rendered = prompt_content_lines_snapshot(&prompt, 42)
        .into_iter()
        .map(|line| {
            line.spans
                .into_iter()
                .map(|span| span.content.into_owned())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    let panel_labels = rendered
        .iter()
        .filter(|line| line.contains("PLAN"))
        .collect::<Vec<_>>();
    assert_eq!(panel_labels.len(), 1);
    assert!(!rendered.iter().any(|line| line.trim() == "Plan"));
}

#[test]
fn interrupted_projection_hides_appended_skill_blocks_in_user_text() {
    let message = lash::Message {
        id: "m1".into(),
        role: lash::MessageRole::User,
        parts: vec![Part {
            id: "m1.p1".into(),
            kind: PartKind::Text,
            content: "Use /wholehog\n\n<skill>\n<name>wholehog</name>\nbody\n</skill>".into(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            prune_state: lash::PruneState::Intact,
        }],
        user_input: Some(lash::UserInputProvenance {
            display_text: "Use /wholehog".into(),
            effective_text: "Use /wholehog\n\n<skill>\n<name>wholehog</name>\nbody\n</skill>"
                .into(),
            transforms: vec![lash::UserInputTransform::SkillBlockAppend {
                skill_name: "wholehog".into(),
                skill_path: "/tmp/wholehog/SKILL.md".into(),
            }],
        }),
        origin: None,
    };

    let blocks =
        projected_blocks_from_state(&[message], &[], &crate::app::UiResumeState::default());

    assert!(matches!(
        blocks.first(),
        Some(DisplayBlock::UserInput(text)) if text == "Use /wholehog"
    ));
    assert!(!blocks.iter().any(|block| match block {
        DisplayBlock::UserInput(text) => text.contains("<skill>") || text.contains("<name>"),
        _ => false,
    }));
}

#[test]
fn snippet_preview_renders_line_numbered_code_block() {
    let preview = SnippetPreviewArtifact {
        title: Some("Queue preview".into()),
        path: "lash-cli/src/render/mod.rs".into(),
        start_line: 12,
        end_line: 13,
        content: "fn one() {}\nfn two() {}".into(),
        render_mode: SnippetRenderMode::Code,
        language: Some("rs".into()),
    };

    let mut rendered = Vec::new();
    render_snippet_preview(&preview, &mut rendered, 80);
    let text = rendered
        .into_iter()
        .map(|line| {
            line.spans
                .into_iter()
                .map(|span| span.content.into_owned())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    assert!(text.iter().any(|line| line.contains("Queue preview")));
    assert!(
        !text
            .iter()
            .any(|line| line.contains("File · lash-cli/src/render/mod.rs:12-13")),
        "metadata line should be suppressed when custom title is present"
    );
    assert!(text.iter().any(|line| line.contains("12 │ fn one() {}")));
    assert!(text.iter().any(|line| line.contains("13 │ fn two() {}")));
}

#[test]
fn activity_block_renders_snippet_preview_at_default_expand_level() {
    let activity = ActivityBlock {
        kind: ActivityKind::GenericTool,
        status: ActivityStatus::Completed,
        tool_name: "showcase_snippet".into(),
        summary: "showcase README.md:1-3".into(),
        detail_lines: Vec::new(),
        duration_ms: 0,
        args: Value::Null,
        result: Value::Null,
        artifact: Some(ActivityArtifact::SnippetPreview(SnippetPreviewArtifact {
            title: Some("README".into()),
            path: "README.md".into(),
            start_line: 1,
            end_line: 3,
            content: "# Title".into(),
            render_mode: SnippetRenderMode::Markdown,
            language: Some("markdown".into()),
        })),
        children: Vec::new(),
        extra: None,
    };

    let blocks = vec![DisplayBlock::Activity(Box::new(activity))];
    let rendered = render_block(&blocks, 0, 1, 80, 24)
        .into_iter()
        .map(|line| {
            line.spans
                .into_iter()
                .map(|span| span.content.into_owned())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    assert!(
        rendered
            .iter()
            .any(|line| line.contains("showcase README.md:1-3"))
    );
    assert!(rendered.iter().any(|line| line.contains("README")));
    assert!(rendered.iter().any(|line| line.contains("Title")));
}

#[test]
fn activity_block_indents_showcase_snippet_preview_under_summary() {
    let activity = ActivityBlock {
        kind: ActivityKind::GenericTool,
        status: ActivityStatus::Completed,
        tool_name: "showcase_snippet".into(),
        summary: "showcase lash/src/plugin_builtin/plan_mode.rs:780-786".into(),
        detail_lines: Vec::new(),
        duration_ms: 0,
        args: Value::Null,
        result: Value::Null,
        artifact: Some(ActivityArtifact::SnippetPreview(SnippetPreviewArtifact {
            title: Some("plan-modes blocked tool message".into()),
            path: "lash/src/plugin_builtin/plan_mode.rs".into(),
            start_line: 780,
            end_line: 786,
            content: "if ctx.tool_name != \"plan_exit\" {\n    return Ok(());\n}".into(),
            render_mode: SnippetRenderMode::Code,
            language: Some("rs".into()),
        })),
        children: Vec::new(),
        extra: None,
    };

    let blocks = vec![DisplayBlock::Activity(Box::new(activity))];
    let rendered = render_block(&blocks, 0, 1, 100, 24)
        .into_iter()
        .map(|line| {
            line.spans
                .into_iter()
                .map(|span| span.content.into_owned())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    assert!(rendered.iter().any(|line| line.starts_with("· showcase lash/src/plugin_builtin/plan_mode.rs:780-786")));
    assert!(
        rendered
            .iter()
            .any(|line| line.starts_with("    plan-modes blocked tool message"))
    );
    assert!(
        !rendered.iter().any(|line| {
            line.starts_with("    File · lash/src/plugin_builtin/plan_mode.rs:780-786")
        }),
        "metadata line should be suppressed when custom title is present"
    );
    assert!(
        rendered
            .iter()
            .any(|line| line.starts_with("    780 │ if ctx.tool_name != \"plan_exit\" {"))
    );
}

#[tokio::test]
async fn showcase_tool_output_wraps_under_line_number_gutter() {
    let temp = tempfile::tempdir().expect("tempdir");
    let path = temp.path().join("sample.rs");
    std::fs::write(
        &path,
        "fn autonomous_prompt_overrides() -> Vec<PromptSectionOverride> {\n",
    )
    .expect("write file");

    let tool = ShowcaseSnippet::new();
    let result = tool
        .execute(
            "showcase_snippet",
            &serde_json::json!({
                "path": path,
                "start_line": 1,
                "end_line": 1,
                "title": "Autonomous-mode prompt overrides"
            }),
        )
        .await;

    assert!(result.success, "tool failed: {:?}", result.result);

    let mut state = ActivityState::default();
    let activity = state
        .blocks_for_tool_call(
            "showcase_snippet",
            serde_json::json!({
                "path": path,
                "start_line": 1,
                "end_line": 1,
                "title": "Autonomous-mode prompt overrides"
            }),
            result.result,
            true,
            5,
        )
        .into_iter()
        .next()
        .expect("activity block");

    let blocks = vec![DisplayBlock::Activity(Box::new(activity))];
    let rendered = render_block(&blocks, 0, 1, 54, 24);

    assert!(rendered.iter().all(|line| line.width() <= 54));

    let text = rendered
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    assert!(
        text.iter()
            .any(|line| line.contains("Autonomous-mode prompt overrides"))
    );
    assert!(
        text.iter()
            .any(|line| line == "     1 │ fn autonomous_prompt_overrides() -> Vec<Promp")
    );
    assert!(
        text.iter()
            .any(|line| line == "       │ tSectionOverride> {")
    );
}

#[test]
fn user_input_does_not_highlight_non_command_slash_word_in_prose() {
    let blocks = vec![DisplayBlock::UserInput(
        "We need to deal with node /relation types.".into(),
    )];

    let rendered = render_block(&blocks, 0, 1, 80, 20);
    let line = rendered
        .iter()
        .find(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
                .contains("/relation")
        })
        .expect("user input line");

    assert!(
        line.spans.iter().any(|span| {
            span.content.contains("/relation") && span.style == theme::user_input()
        })
    );
}

#[test]
fn shell_activity_renders_live_output_inline_under_tool() {
    let mut app = App::new("test-model".into(), "test".into());
    app.blocks = vec![DisplayBlock::Activity(Box::new(ActivityBlock {
        kind: ActivityKind::ShellCommand,
        status: ActivityStatus::Completed,
        tool_name: "exec_command".into(),
        summary: "started cargo check".into(),
        detail_lines: vec!["Handle shell-1".into()],
        duration_ms: 0,
        args: Value::Null,
        result: Value::Null,
        artifact: None,
        children: Vec::new(),
        extra: None,
    }))];
    let now = std::time::Instant::now();
    app.running = true;
    app.live_turn = Some(crate::app::LiveTurnState {
        status_text: "shell".into(),
        status_detail: None,
        phase_started_at: now,
        turn_started_at: now,
        has_visible_output: true,
        output_start_anchor_pending: false,
        transient_until: None,
    });
    app.streaming_output = vec!["Compiling lash-cli".into(), "warning: unused import".into()];

    let rendered = app
        .rendered_block_lines_cached(0, 64, 20)
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    assert!(
        rendered
            .iter()
            .any(|line| line.contains("started cargo check"))
    );
    assert!(rendered.iter().any(|line| line.contains("Handle shell-1")));
    assert!(
        rendered
            .iter()
            .any(|line| line.contains("Compiling lash-cli"))
    );
    assert!(
        rendered
            .iter()
            .any(|line| line.contains("warning: unused import"))
    );
}

#[test]
fn plugin_panel_renders_as_section_header_without_box() {
    let blocks = vec![DisplayBlock::PluginPanel(crate::app::PluginPanelBlock {
        plugin_id: "plan_mode".into(),
        key: "panel".into(),
        title: "PLAN".into(),
        content: "Entered plan mode.\n\n`.lash/plans/demo.md`".into(),
    })];

    let rendered = render_block(&blocks, 0, 1, 72, 20)
        .into_iter()
        .map(|line| {
            line.spans
                .into_iter()
                .map(|span| span.content.into_owned())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    assert!(rendered.iter().any(|line| line.contains("PLAN")));
    assert!(
        rendered
            .iter()
            .any(|line| line.contains("Entered plan mode."))
    );
    assert!(
        rendered
            .iter()
            .any(|line| line.contains(".lash/plans/demo.md"))
    );
    assert!(!rendered.iter().any(|line| line.contains("┌")));
    assert!(!rendered.iter().any(|line| line.contains("└")));
}

#[test]
fn styled_snippet_chunk_highlights_code_tokens() {
    let spans = artifact::styled_snippet_chunk_for_test(
        "12 │ fn main() { let name = \"sam\"; // note",
        theme::code_content(),
        Some("rs"),
    );

    assert!(
        spans
            .iter()
            .any(|span| span.content.contains("fn") && span.style == theme::code_keyword())
    );
    assert!(
        spans
            .iter()
            .any(|span| span.content.contains("\"sam\"") && span.style == theme::code_string())
    );
    assert!(
        spans
            .iter()
            .any(|span| span.content.contains("// note") && span.style == theme::code_comment())
    );
}

#[test]
fn snippet_preview_preserves_markdown_styling() {
    let preview = SnippetPreviewArtifact {
        title: Some("Docs".into()),
        path: "README.md".into(),
        start_line: 1,
        end_line: 3,
        content: "# Heading\n\n`inline`".into(),
        render_mode: SnippetRenderMode::Markdown,
        language: Some("markdown".into()),
    };

    let mut rendered = Vec::new();
    render_snippet_preview(&preview, &mut rendered, 80);

    let heading_line = rendered
        .iter()
        .find(|line| {
            line.spans
                .iter()
                .any(|span| span.content.contains("Heading"))
        })
        .expect("heading line");
    assert!(
        heading_line
            .spans
            .iter()
            .any(|span| span.content.contains("Heading") && span.style == theme::heading())
    );

    let inline_code_line = rendered
        .iter()
        .find(|line| {
            line.spans
                .iter()
                .any(|span| span.content.contains("inline"))
        })
        .expect("inline code line");
    assert!(
        inline_code_line
            .spans
            .iter()
            .any(|span| span.content.contains("inline") && span.style == theme::inline_code())
    );
}

#[test]
fn snippet_preview_renders_markdown_list_snippet_without_literal_emphasis_markers() {
    let preview = SnippetPreviewArtifact {
        title: Some("README".into()),
        path: "README.md".into(),
        start_line: 11,
        end_line: 14,
        content: "- **Two execution modes**\n  - `repl` (default) — runs a persistent `lashlang` DSL runtime.\n  - `standard` — uses the provider's native tool-calling protocol directly.".into(),
        render_mode: SnippetRenderMode::Markdown,
        language: Some("markdown".into()),
    };

    let mut rendered = Vec::new();
    render_snippet_preview(&preview, &mut rendered, 100);
    let text = rendered
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    assert!(text.iter().any(|line| line.contains("Two execution modes")));
    assert!(
        !text
            .iter()
            .any(|line| line.contains("**Two execution modes**"))
    );
    assert!(text.iter().any(|line| line.contains("repl")));
    assert!(text.iter().any(|line| line.contains("standard")));
}

#[test]
fn snippet_preview_wraps_long_markdown_bullets_to_viewport_width() {
    let preview = SnippetPreviewArtifact {
        title: Some("Plan excerpt".into()),
        path: ".lash/plans/demo.md".into(),
        start_line: 1,
        end_line: 4,
        content: "## Goal\n\n- Complete a full spring-cleaning pass in two phases: first remove dead or stale things, then simplify what remains.".into(),
        render_mode: SnippetRenderMode::Markdown,
        language: Some("markdown".into()),
    };

    let mut rendered = Vec::new();
    render_snippet_preview(&preview, &mut rendered, 48);

    assert!(rendered.iter().all(|line| line.width() <= 48));

    let text = rendered
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>();
    assert!(text.iter().any(|line| line.contains("Complete a full")));
    assert!(text.iter().any(|line| line.contains("spring-cleaning")));
}
