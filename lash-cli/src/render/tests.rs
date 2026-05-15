use super::artifact::render_snippet_preview;
use super::prompt::prompt_content_lines_snapshot;
use super::*;
use crate::SkillCatalog;
use crate::activity::ActivityState;
use crate::app::timeline_from_read_view;
use crate::assistant_text::normalize_assistant_text;
use crate::theme;
use async_trait::async_trait;
use lash_core::{Part, PartKind, PromptRequest};
use lash_tui_extensions::{
    SlashCommandSpec, TuiExtension, TuiExtensionContext, TuiExtensions, TuiHostEffect,
};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;

fn timeline_items_from_test_read_view(
    events: &[lash_core::SessionEventRecord],
    messages: &[lash_core::Message],
    tool_calls: &[lash_core::ToolCallRecord],
    ui_state: &crate::app::UiProjectionState,
) -> Vec<crate::app::UiTimelineItem> {
    let read_view = lash_core::SessionReadView::from_derived_message_view(
        lash_core::SessionStateEnvelope::default(),
        Arc::new(events.to_vec()),
        Arc::new(messages.to_vec()),
        Arc::new(tool_calls.to_vec()),
    );
    timeline_from_read_view(&read_view, ui_state)
        .items()
        .to_vec()
}

fn skill_catalog_with(names: &[(&str, &str)]) -> SkillCatalog {
    let root = std::env::temp_dir().join(format!("lash-render-skills-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).expect("temp root");
    for (name, description) in names {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).expect("skill dir");
        std::fs::write(
            dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: {description}\n---\n\nbody\n"),
        )
        .expect("skill file");
    }
    let catalog = SkillCatalog::from_dirs(&[PathBuf::from(&root)]);
    let _ = std::fs::remove_dir_all(root);
    catalog
}

fn skill_catalog_with_hints(entries: &[(&str, &str, Option<&str>)]) -> SkillCatalog {
    let root =
        std::env::temp_dir().join(format!("lash-render-skills-hints-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).expect("temp root");
    for (name, description, argument_hint) in entries {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).expect("skill dir");
        let hint_line = argument_hint
            .map(|hint| format!("argument-hint: \"{hint}\"\n"))
            .unwrap_or_default();
        std::fs::write(
            dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: {description}\n{hint_line}---\n\nbody\n"),
        )
        .expect("skill file");
    }
    let catalog = SkillCatalog::from_dirs(&[PathBuf::from(&root)]);
    let _ = std::fs::remove_dir_all(root);
    catalog
}

#[test]
fn exploration_multi_op_shows_explored_header_with_ops_below() {
    // Multi-op exploration: "Explored" header + op list indented below.
    // No step counter — the list makes the count redundant.
    let activity = ActivityBlock::new(
        ActivityKind::Exploration,
        "search",
        Value::Null,
        "Explored",
        ActivityStatus::Completed,
        Value::Null,
        0,
    )
    .with_detail_lines(vec!["Read README.md".into(), "Read Cargo.toml".into()]);
    let blocks = vec![UiTimelineItem::Activity(Box::new(activity))];
    let rendered = render_block(&blocks, 0, 1, 80, 20)
        .into_iter()
        .map(|line| {
            line.spans
                .into_iter()
                .map(|span| span.content.into_owned())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    assert_eq!(rendered[0], "• Explored");
    assert_eq!(rendered[1], "    Read README.md");
    assert_eq!(rendered[2], "    Read Cargo.toml");
}

#[test]
fn single_op_exploration_renders_as_one_line() {
    // Solo exploration tool calls (1 read, 1 grep, etc.) render as a
    // single top-level bullet — no `EXPLORE` wrapper, no `1 step` body,
    // no indented detail line duplicating the header.
    let mut state = crate::activity::ActivityState::default();
    let blocks = state.project_tool_call(
        "read_file",
        serde_json::json!({ "path": "README.md" }),
        Value::Null,
        true,
        0,
    );
    let display_blocks = vec![UiTimelineItem::Activity(Box::new(blocks[0].clone()))];
    let rendered = render_block(&display_blocks, 0, 1, 80, 20)
        .into_iter()
        .map(|line| {
            line.spans
                .into_iter()
                .map(|span| span.content.into_owned())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    assert_eq!(rendered[0], "• Read README.md");
    assert_eq!(rendered.len(), 1, "single-op should not emit detail lines");
}

#[test]
fn subagent_headline_stays_compact_and_task_wraps_in_detail_rows() {
    use lash_subagents::AgentMetadata;
    let host = std::sync::Arc::new(crate::activity::tests::MockSubagentHost::default());
    host.insert(
        "root",
        "probe_repo_shape",
        AgentMetadata {
            session_id: "child".to_string(),
            parent_session_id: Some("root".to_string()),
            capability: Some("explore".to_string()),
            state: "running".to_string(),
            model: "gpt-5.4-mini".to_string(),
            model_variant: Some("low".to_string()),
            last_iterations: None,
            last_tool_calls: None,
            last_token_usage: None,
        },
    );
    let mut state = crate::activity::ActivityState::default();
    state.set_subagent_host(host);
    let blocks = state.project_tool_call(
        "spawn_agent",
        serde_json::json!({
            "agent_name":"probe_repo_shape",
            "task":"In /home/sam/code/lash, inspect the repo shape only. Reply with 1) top-level directories/files summary and 2) whether the workspace looks healthy.",
            "capability":"explore"
        }),
        serde_json::json!({
            "agent_name":"probe_repo_shape",
            }),
        true,
        0,
    );
    let display_blocks = vec![UiTimelineItem::Activity(Box::new(blocks[0].clone()))];
    let rendered = render_block(&display_blocks, 0, 1, 56, 20)
        .into_iter()
        .map(|line| {
            line.spans
                .into_iter()
                .map(|span| span.content.into_owned())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    assert!(rendered.iter().all(|line| line.chars().count() <= 56));
    assert_eq!(rendered[0], "◆ spawn subagent · probe_repo_shape");
    assert!(
        rendered
            .iter()
            .any(|line| line.starts_with("    Task In /home/sam/code/lash,"))
    );
    assert!(
        rendered
            .iter()
            .any(|line| line.contains("workspace looks healthy."))
    );
    assert!(
        rendered
            .iter()
            .any(|line| line == "    Agent probe_repo_shape")
    );
    assert!(
        rendered
            .iter()
            .any(|line| line == "    Profile explore capability")
    );
}

#[test]
fn extract_history_selection_text_reads_across_scrolled_content_rows() {
    let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
    app.timeline = vec![UiTimelineItem::UserInput("alpha\nbeta\ngamma".into())].into();
    app.scroll_offset = 1;
    app.selection.anchor = (2, 1);
    app.selection.end = (4, 2);
    app.selection.visible = true;

    let selected = extract_history_selection_text(&app, 40, 9).expect("selected text");
    assert_eq!(selected, "beta\n  ga");
}

#[test]
fn input_byte_offset_mapping_tracks_wrapped_rows() {
    assert_eq!(input_byte_offset_at_visual_position("abcdef", 1, 3, 6), 5);
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
    let message = lash_core::Message {
        id: "m1".into(),
        role: lash_core::MessageRole::User,
        parts: vec![Part {
            id: "m1.p1".into(),
            kind: PartKind::Text,
            content: "Use /wholehog\n\n<skill>\n<name>wholehog</name>\nbody\n</skill>".into(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: lash_core::PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]
        .into(),
        origin: None,
    };

    let blocks = timeline_items_from_test_read_view(
        &[],
        &[message],
        &[],
        &crate::app::UiProjectionState::default(),
    );

    // blocks[0] is the TurnStart marker emitted before the user input.
    assert!(matches!(blocks.first(), Some(UiTimelineItem::TurnStart(_))));
    assert!(matches!(
        blocks.get(1),
        Some(UiTimelineItem::UserInput(text)) if text.contains("Use /wholehog")
    ));
}

#[test]
fn assistant_text_keeps_literal_repl_tags_in_prose() {
    let text = "Use the literal tag <rlm> and then close </rlm>.";
    assert_eq!(normalize_assistant_text(text), text);
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
    let activity = ActivityBlock::new(
        ActivityKind::GenericTool,
        "preview_text",
        Value::Null,
        "preview README.md:1-3",
        ActivityStatus::Completed,
        Value::Null,
        0,
    )
    .with_artifact(Some(ActivityArtifact::SnippetPreview(
        SnippetPreviewArtifact {
            title: Some("README".into()),
            path: "README.md".into(),
            start_line: 1,
            end_line: 3,
            content: "# Title".into(),
            render_mode: SnippetRenderMode::Markdown,
            language: Some("markdown".into()),
        },
    )));

    let blocks = vec![UiTimelineItem::Activity(Box::new(activity))];
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
            .any(|line| line.contains("preview README.md:1-3"))
    );
    assert!(rendered.iter().any(|line| line.contains("README")));
    assert!(rendered.iter().any(|line| line.contains("Title")));
}

#[test]
fn activity_block_indents_snippet_preview_under_summary() {
    let activity = ActivityBlock::new(
        ActivityKind::GenericTool,
        "preview_text",
        Value::Null,
        "preview lash/src/plugin_builtin/plan_mode.rs:780-786",
        ActivityStatus::Completed,
        Value::Null,
        0,
    )
    .with_artifact(Some(ActivityArtifact::SnippetPreview(
        SnippetPreviewArtifact {
            title: Some("plan-modes blocked tool message".into()),
            path: "lash/src/plugin_builtin/plan_mode.rs".into(),
            start_line: 780,
            end_line: 786,
            content: "if ctx.tool_name != \"plan_exit\" {\n    return Ok(());\n}".into(),
            render_mode: SnippetRenderMode::Code,
            language: Some("rs".into()),
        },
    )));

    let blocks = vec![UiTimelineItem::Activity(Box::new(activity))];
    let rendered = render_block(&blocks, 0, 1, 100, 24)
        .into_iter()
        .map(|line| {
            line.spans
                .into_iter()
                .map(|span| span.content.into_owned())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    assert!(rendered.iter().any(|line| {
        line.starts_with("• preview lash/src/plugin_builtin/plan_mode.rs:780-786")
    }));
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

#[test]
fn user_input_does_not_highlight_non_command_slash_word_in_prose() {
    let blocks = vec![UiTimelineItem::UserInput(
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
fn user_input_highlights_slash_in_leading_slash_command() {
    let blocks = vec![UiTimelineItem::UserInput("/help with something".into())];

    let rendered = render_block(&blocks, 0, 1, 80, 20);
    let line = rendered.first().expect("user input line");

    assert!(
        line.spans.iter().any(|span| {
            span.content.contains('/') && span.style == theme::slash_command_slash()
        })
    );
    assert!(
        line.spans
            .iter()
            .any(|span| span.content.contains("help with something")
                && span.style == theme::user_input())
    );
}

#[test]
fn input_box_highlights_slash_command_slash() {
    let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
    app.set_input("/model gpt-5.4".into());

    let snapshot = input_render_snapshot(&app, Rect::new(0, 0, 40, 4));
    let first_line = snapshot.lines.first().expect("input line");

    assert!(
        first_line.spans.iter().any(|span| {
            span.content.contains('/') && span.style == theme::slash_command_slash()
        })
    );
}

#[test]
fn input_box_shows_skill_argument_hint_inline() {
    let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
    app.skills =
        skill_catalog_with_hints(&[("impeccable", "design helper", Some("[craft|teach|extract]"))]);
    app.set_input("/impeccable ".into());
    app.editor.cursor_pos = app.input().len();

    let snapshot = input_render_snapshot(&app, Rect::new(0, 0, 60, 4));
    let first_line = snapshot.lines.first().expect("input line");
    let rendered = first_line
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();

    assert!(rendered.contains("/impeccable [craft|teach|extract]"));
    assert!(first_line.spans.iter().any(|span| {
        span.content.contains("[craft|teach|extract]") && span.style == theme::text_faint_style()
    }));
}

#[test]
fn input_box_shows_ui_command_argument_hint_inline() {
    struct DemoTuiExtension;

    const DEMO_COMMANDS: &[SlashCommandSpec] = &[SlashCommandSpec {
        name: "/demo",
        aliases: &[],
        usage: "/demo [alpha|beta]",
        description: "Demo command",
        argument_hint: Some("[alpha|beta]"),
        argument_options: &["alpha", "beta"],
        takes_argument: true,
        allow_while_running: true,
        action: "demo",
    }];

    #[async_trait]
    impl TuiExtension for DemoTuiExtension {
        fn id(&self) -> &'static str {
            "demo_ui"
        }

        fn commands(&self) -> &'static [SlashCommandSpec] {
            DEMO_COMMANDS
        }

        async fn invoke_action(
            &self,
            _action: &str,
            _arg: Option<&str>,
            _ctx: TuiExtensionContext<'_>,
        ) -> Result<Vec<TuiHostEffect>, String> {
            Ok(Vec::new())
        }
    }

    let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
    let ui_extensions =
        TuiExtensions::new(vec![Arc::new(DemoTuiExtension)]).expect("ui extensions");
    app.set_ui_extensions(Arc::new(ui_extensions));
    app.set_input("/demo ".into());
    app.editor.cursor_pos = app.input().len();

    let snapshot = input_render_snapshot(&app, Rect::new(0, 0, 40, 4));
    let first_line = snapshot.lines.first().expect("input line");
    let rendered = first_line
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();

    assert!(rendered.contains("/demo [alpha|beta]"));
    assert!(first_line.spans.iter().any(|span| {
        span.content.contains("[alpha|beta]") && span.style == theme::text_faint_style()
    }));
}

#[test]
fn queue_preview_highlights_slash_command_slash() {
    let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
    let turn = PreparedTurn::prepare("/retry later".into(), Vec::new(), &app.skills);
    app.queue_turn(turn);

    let rendered = queue_preview_lines_snapshot(&app, 40);
    let item_line = rendered
        .iter()
        .find(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
                .contains("/retry later")
        })
        .expect("queue preview line");

    assert!(
        item_line.spans.iter().any(|span| {
            span.content.contains('/') && span.style == theme::slash_command_slash()
        })
    );
}

#[test]
fn user_input_highlights_every_detected_slash_command() {
    let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
    app.skills = skill_catalog_with(&[
        ("spring-cleaning", "cleanup"),
        ("yolopush", "ship changes"),
        ("ghmonitor", "monitor"),
    ]);
    app.timeline = vec![UiTimelineItem::UserInput(
        "/spring-cleaning /yolopush and then /ghmonitor status".into(),
    )]
    .into();

    let rendered = render_block_lines(&app, 0, 80, 20);
    let slash_spans = rendered
        .iter()
        .flat_map(|line| line.spans.iter())
        .filter(|span| span.content == "/" && span.style == theme::slash_command_slash())
        .count();

    assert_eq!(slash_spans, 3);
}

#[test]
fn user_input_does_not_highlight_unknown_inline_slash_words() {
    let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
    app.skills = skill_catalog_with(&[("ghmonitor", "monitor")]);
    app.timeline = vec![UiTimelineItem::UserInput(
        "Please check /not-a-command and /ghmonitor soon".into(),
    )]
    .into();

    let rendered = render_block_lines(&app, 0, 80, 20);
    let line = rendered.first().expect("user input line");

    assert!(
        line.spans
            .iter()
            .any(|span| { span.content == "/" && span.style == theme::slash_command_slash() })
    );
    assert!(line.spans.iter().any(|span| {
        span.content.contains("/not-a-command") && span.style == theme::user_input()
    }));
}

#[test]
fn queue_preview_highlights_multiple_detected_slash_commands() {
    let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
    app.skills = skill_catalog_with(&[("ghmonitor", "monitor")]);
    let turn = PreparedTurn::prepare(
        "/retry then /ghmonitor and /bogus".into(),
        Vec::new(),
        &app.skills,
    );
    app.queue_turn(turn);

    let rendered = queue_preview_lines_snapshot(&app, 80);
    let slash_spans = rendered
        .iter()
        .flat_map(|line| line.spans.iter())
        .filter(|span| span.content == "/" && span.style == theme::slash_command_slash())
        .count();

    assert_eq!(slash_spans, 2);
}

#[test]
fn shell_activity_renders_live_output_inline_under_tool() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline = vec![UiTimelineItem::Activity(Box::new(
        ActivityBlock::new(
            ActivityKind::ShellCommand,
            "exec_command",
            Value::Null,
            "started cargo check",
            ActivityStatus::Running,
            Value::Null,
            0,
        )
        .with_detail_lines(vec!["Handle shell-1".into()]),
    ))]
    .into();
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
    app.live_tool_output.lines = vec!["Compiling lash-cli".into(), "warning: unused import".into()];

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
fn live_tool_output_without_running_activity_renders_as_tail_block() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline = vec![UiTimelineItem::Activity(Box::new(ActivityBlock::new(
        ActivityKind::ShellCommand,
        "exec_command",
        Value::Null,
        "pwd",
        ActivityStatus::Completed,
        Value::Null,
        0,
    )))]
    .into();
    app.live_tool_output.title = Some("cargo check -p lash-cli".into());
    app.live_tool_output.lines = vec!["Checking lash v0.2.91".into()];

    let completed_block = app
        .rendered_block_lines_cached(0, 64, 20)
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>();
    assert!(completed_block.iter().any(|line| line.contains("pwd")));
    assert!(
        !completed_block
            .iter()
            .any(|line| line.contains("Checking lash"))
    );

    let tail = live_tool_output_standalone_lines(&app, 64)
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>();
    assert!(
        tail.iter()
            .any(|line| line.contains("cargo check -p lash-cli"))
    );
    assert!(tail.iter().any(|line| line.contains("Checking lash")));
}

#[test]
fn plugin_panel_renders_as_section_header_without_box() {
    let blocks = vec![UiTimelineItem::PluginPanel(crate::app::PluginPanelBlock {
        plugin_id: "plan_mode".into(),
        key: "panel".into(),
        title: "PLAN".into(),
        content: "Path: `.lash/plans/demo.md`".into(),
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
            .any(|line| line.contains("Path: .lash/plans/demo.md"))
    );
    assert!(!rendered.iter().any(|line| line.contains("┌")));
    assert!(!rendered.iter().any(|line| line.contains("└")));
}

#[test]
fn plan_dock_renders_as_checklist_with_dim_plan_header() {
    use crate::app::{App, PlanDockItem, PlanDockItemStatus, PlanDockState};
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.plan_dock = Some(PlanDockState {
        title: "PLAN".into(),
        meta: None,
        items: vec![
            PlanDockItem {
                text: "already done".into(),
                status: PlanDockItemStatus::Done,
            },
            PlanDockItem {
                text: "in flight".into(),
                status: PlanDockItemStatus::Active,
            },
            PlanDockItem {
                text: "not yet".into(),
                status: PlanDockItemStatus::Pending,
            },
        ],
    });

    let lines = crate::render::plan_dock_lines_snapshot(&app, 80)
        .expect("plan dock should render when items are present");
    let text: Vec<String> = lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect();

    // Header row + 3 items = 4 rows. No scribe rule.
    assert_eq!(text.len(), 4, "expected 1 header + 3 items, got {text:?}");
    assert!(text[0].contains("Plan"));
    assert!(text[1].contains("✓") && text[1].contains("already done"));
    assert!(text[2].contains("▶") && text[2].contains("in flight"));
    assert!(text[3].contains("□") && text[3].contains("not yet"));
    assert!(
        !text.iter().any(|line| line.contains("─")),
        "no scribe rule expected, got {text:?}",
    );
}

#[test]
fn plan_dock_trailing_height_includes_gutter_plus_items() {
    use crate::app::{App, PlanDockItem, PlanDockItemStatus, PlanDockState};
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    assert_eq!(crate::render::plan_dock_trailing_height(&app), 0);

    app.plan_dock = Some(PlanDockState {
        title: String::new(),
        meta: None,
        items: vec![
            PlanDockItem {
                text: "a".into(),
                status: PlanDockItemStatus::Pending,
            },
            PlanDockItem {
                text: "b".into(),
                status: PlanDockItemStatus::Pending,
            },
        ],
    });
    assert_eq!(crate::render::plan_dock_trailing_height(&app), 3);
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
        content: "- **Two execution modes**\n  - `rlm` (default) — runs a persistent `lashlang` DSL runtime.\n  - `standard` — uses the provider's native tool-calling protocol directly.".into(),
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
    assert!(text.iter().any(|line| line.contains("rlm")));
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

#[test]
fn lashlang_code_block_is_hidden_below_full_expand() {
    let blocks = vec![UiTimelineItem::LashlangCode(
        "r = call read_file { path: \"a\" }\nsubmit r.value".to_string(),
    )];
    for level in [0u8, 1] {
        let rendered = render_block(&blocks, 0, level, 80, 24);
        assert!(
            rendered.is_empty(),
            "expected no output at expand_level {level}, got {rendered:?}",
        );
    }
}

// ── Expansion-level mapping (see docs/design-language.html) ──────────
//
// L0 (default)   — summary · detail lines · QuestionPanel · compact
//                  patch (no diffs) · snippet preview.
// L1 (Ctrl+O)    — above · inline patch diffs · shell output.
// L2 (Alt+O)     — above · DiffPreview/TextPreview/SourceList · full
//                  reasoning · uncapped patch diffs · lashlang code.

#[test]
fn activity_detail_lines_are_visible_at_l0() {
    let mut state = ActivityState::default();
    let blocks = state.project_tool_call(
        "monitor",
        serde_json::json!({
            "description": "build",
            "command": "cargo build",
        }),
        serde_json::json!({
            "description": "build",
            "command": "cargo build",
            "persistent": false,
            "timeout_ms": 300000,
            "run_state": "running",
        }),
        true,
        3,
    );
    let display = vec![UiTimelineItem::Activity(Box::new(blocks[0].clone()))];
    let rendered = render_block(&display, 0, 0, 80, 10);
    let text: Vec<String> = rendered
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect();
    assert!(
        text.iter().any(|line| line.contains("running · build")),
        "detail lines should render at L0; got {text:?}",
    );
}

#[test]
fn shell_output_is_hidden_at_l0_and_visible_at_l1() {
    let blocks = vec![UiTimelineItem::ShellOutput {
        command: "echo hi".into(),
        output: "hi\nworld".into(),
        error: None,
    }];

    let l0 = render_block(&blocks, 0, 0, 40, 10);
    let l0_text: Vec<String> = l0
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect();
    assert!(
        !l0_text.iter().any(|line| line.contains("world")),
        "shell body must stay off at L0; got {l0_text:?}",
    );

    let l1 = render_block(&blocks, 0, 1, 40, 10);
    let l1_text: Vec<String> = l1
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect();
    assert!(
        l1_text.iter().any(|line| line.contains("world")),
        "shell body must render at L1; got {l1_text:?}",
    );
}

#[test]
fn reasoning_is_compact_below_l2_and_full_at_l2() {
    let blocks = vec![
        UiTimelineItem::UserInput("x".into()),
        UiTimelineItem::AssistantReasoning("**Planning the push**\n\nThinking body line.".into()),
        UiTimelineItem::AssistantText("Answer.".into()),
    ];

    let l0 = render_block(&blocks, 1, 0, 60, 10);
    let l0_text: Vec<String> = l0
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect();
    assert!(
        l0_text
            .iter()
            .any(|line| line.contains("Planning the push")),
        "L0 reasoning should show compact preview; got {l0_text:?}",
    );
    assert!(
        !l0_text.iter().any(|line| line.contains("alt+o")),
        "L0 reasoning should not repeat expansion key hints; got {l0_text:?}",
    );
    assert!(
        !l0_text
            .iter()
            .any(|line| line.contains("Thinking body line")),
        "L0 must hide the reasoning body; got {l0_text:?}",
    );

    let l1 = render_block(&blocks, 1, 1, 60, 10);
    let l1_text: Vec<String> = l1
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect();
    assert!(
        !l1_text
            .iter()
            .any(|line| line.contains("Thinking body line")),
        "L1 must still hide the reasoning body; got {l1_text:?}",
    );

    let l2 = render_block(&blocks, 1, 2, 60, 10);
    let l2_text: Vec<String> = l2
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect();
    assert!(
        l2_text
            .iter()
            .any(|line| line.contains("Thinking body line")),
        "L2 should render the reasoning body; got {l2_text:?}",
    );
}

#[test]
fn live_reasoning_compacts_after_activity_appends_below_it() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();
    app.timeline = vec![UiTimelineItem::AssistantReasoning(
        "**Inspecting config implementation**\n\nDetailed thinking body.".into(),
    )]
    .into();

    app.expand_level = 1;
    let live_text: Vec<String> = app
        .rendered_block_lines_cached(0, 80, 24)
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect();
    assert!(
        live_text
            .iter()
            .any(|line| line.contains("Detailed thinking body")),
        "live reasoning should expand while it is the tail; got {live_text:?}",
    );

    app.handle_session_event(lash_core::SessionEvent::ToolCall {
        call_id: None,
        name: "read_file".into(),
        args: serde_json::json!({ "path": "lash/src/provider.rs" }),
        output: lash_core::ToolCallOutput::success(
            serde_json::json!({ "content": "provider code" }),
        ),
        duration_ms: 0,
    });

    let compact_text: Vec<String> = app
        .rendered_block_lines_cached(0, 80, 24)
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect();
    assert!(
        compact_text
            .iter()
            .any(|line| line.contains("Inspecting config implementation")),
        "reasoning should keep its compact preview after another block appends; got {compact_text:?}",
    );
    assert!(
        !compact_text
            .iter()
            .any(|line| line.contains("Detailed thinking body")),
        "reasoning body should compact after it is no longer the live tail; got {compact_text:?}",
    );
}

#[test]
fn committed_reasoning_compacts_while_live_assistant_streams() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();
    app.expand_level = 1;
    app.timeline = vec![UiTimelineItem::AssistantReasoning(
        "**Inspecting config implementation**\n\nDetailed thinking body.".into(),
    )]
    .into();

    app.handle_session_event(lash_core::SessionEvent::TextDelta {
        content: "Answer is streaming.".into(),
    });

    let compact_text: Vec<String> = app
        .rendered_block_lines_cached(0, 80, 24)
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect();
    assert!(
        compact_text
            .iter()
            .any(|line| line.contains("Inspecting config implementation")),
        "reasoning should keep its compact preview; got {compact_text:?}",
    );
    assert!(
        !compact_text
            .iter()
            .any(|line| line.contains("Detailed thinking body")),
        "reasoning body should compact once live assistant text exists; got {compact_text:?}",
    );
}

#[test]
fn live_reasoning_compacts_after_turn_stops() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();
    app.timeline = vec![UiTimelineItem::AssistantReasoning(
        "**Inspecting config implementation**\n\nDetailed thinking body.".into(),
    )]
    .into();

    let live_text: Vec<String> = app
        .rendered_block_lines_cached(0, 80, 24)
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect();
    assert!(
        live_text
            .iter()
            .any(|line| line.contains("Detailed thinking body")),
        "live reasoning should expand while a turn is running; got {live_text:?}",
    );

    app.stop_turn();

    let compact_text: Vec<String> = app
        .rendered_block_lines_cached(0, 80, 24)
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect();
    assert!(
        compact_text
            .iter()
            .any(|line| line.contains("Inspecting config implementation")),
        "stopped reasoning should keep its compact preview; got {compact_text:?}",
    );
    assert!(
        !compact_text
            .iter()
            .any(|line| line.contains("Detailed thinking body")),
        "reasoning body should compact after the turn stops; got {compact_text:?}",
    );
}

#[test]
fn lashlang_code_block_renders_header_and_body_at_full_expand() {
    let code = "r = call read_file { path: \"a\" }\nsubmit r.value";
    let blocks = vec![UiTimelineItem::LashlangCode(code.to_string())];
    let rendered = render_block(&blocks, 0, 2, 80, 24);
    let text: Vec<String> = rendered
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect();
    // Header line and two body lines with the `╎ ` gutter.
    assert!(
        text.iter().any(|line| line == "lashlang"),
        "missing header in {text:?}",
    );
    assert!(
        text.iter()
            .any(|line| line.starts_with("╎ r = call read_file")),
        "missing first code line in {text:?}",
    );
    assert!(
        text.iter().any(|line| line.starts_with("╎ submit r.value")),
        "missing submit line in {text:?}",
    );
}
