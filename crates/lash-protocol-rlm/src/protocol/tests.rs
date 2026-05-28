use super::fence::extract_first_lashlang_fence;
use super::*;

#[test]
fn rlm_execution_section_default_prompt_is_golden() {
    insta::with_settings!({ snapshot_path => "../snapshots" }, {
        insta::assert_snapshot!(
            "rlm_execution_section_default",
            rlm_execution_section_for_surface(RlmPromptFeatures::default(), &full_prompt_surface())
        );
    });
}

#[test]
fn rlm_execution_section_no_images_prompt_is_golden() {
    insta::with_settings!({ snapshot_path => "../snapshots" }, {
        insta::assert_snapshot!(
            "rlm_execution_section_no_images",
            rlm_execution_section_for_surface(
                RlmPromptFeatures {
                    images: false,
                    ..RlmPromptFeatures::default()
                },
                &full_prompt_surface()
            )
        );
    });
}

fn prompt_surface(
    resources: lashlang::ResourceCatalog,
    abilities: lashlang::LashlangAbilities,
) -> lashlang::LashlangSurface {
    lashlang::LashlangSurface::new(resources, abilities)
}

fn tool_resources() -> lashlang::ResourceCatalog {
    lashlang::ResourceCatalog::tool_default(["tool_name", "tool_a", "tool_b"])
}

fn full_prompt_surface() -> lashlang::LashlangSurface {
    prompt_surface(tool_resources(), lashlang::LashlangAbilities::all())
}

#[test]
fn execution_section_hides_processes_when_disabled() {
    let surface = prompt_surface(tool_resources(), lashlang::LashlangAbilities::default());
    let section = rlm_execution_section_for_surface(RlmPromptFeatures::default(), &surface);

    assert!(!section.contains("process name"));
    assert!(!section.contains("start name"));
    assert!(!section.contains("sleep for"));
    assert!(!section.contains("wait signal"));
    assert!(!section.contains("signal run"));
}

#[test]
fn execution_section_hides_process_sleep_and_signals_independently() {
    let surface = prompt_surface(
        tool_resources(),
        lashlang::LashlangAbilities::default().with_processes(),
    );
    let section = rlm_execution_section_for_surface(RlmPromptFeatures::default(), &surface);

    assert!(section.contains("process name"));
    assert!(!section.contains("sleep for"));
    assert!(!section.contains("wait signal"));
    assert!(!section.contains("signal run"));
}

#[test]
fn execution_section_hides_trigger_and_schedule_language_when_disabled() {
    let surface = prompt_surface(
        tool_resources(),
        lashlang::LashlangAbilities::default()
            .with_processes()
            .with_process_lifecycle(),
    );
    let section = rlm_execution_section_for_surface(RlmPromptFeatures::default(), &surface);

    assert!(!section.contains("trigger name"));
    assert!(!section.contains("schedule name"));
    assert!(!section.contains("cron("));
    assert!(!section.contains("triggers, or schedules"));
}

#[test]
fn execution_section_hides_trigger_language_without_processes() {
    let surface = prompt_surface(
        tool_resources(),
        lashlang::LashlangAbilities::default().with_triggers(),
    );
    let section = rlm_execution_section_for_surface(RlmPromptFeatures::default(), &surface);

    assert!(!section.contains("trigger name"));
    assert!(!section.contains("resource-event declarations"));
}

#[test]
fn execution_section_lists_exposed_host_event_resources_with_formatted_payloads() {
    let mut resources = lashlang::ResourceCatalog::new();
    resources.add_alias("TRIGGER", "button");
    resources.add_trigger_event(
        "TRIGGER",
        "pressed",
        lashlang::TypeExpr::Object(vec![
            lashlang::TypeField {
                name: "button".into(),
                ty: lashlang::TypeExpr::Enum(vec!["Red".into(), "Blue".into()]),
                optional: false,
            },
            lashlang::TypeField {
                name: "pressed_at".into(),
                ty: lashlang::TypeExpr::Str,
                optional: false,
            },
        ]),
    );
    let surface = prompt_surface(
        resources,
        lashlang::LashlangAbilities::default()
            .with_processes()
            .with_triggers(),
    );

    let section = rlm_execution_section_for_surface(RlmPromptFeatures::default(), &surface);

    assert!(section.contains("### Host Events"));
    assert!(section.contains("`TRIGGER.button.pressed`"));
    assert!(section.contains(r#"{ button: enum["Red", "Blue"], pressed_at: str }"#));
    assert!(section.contains("host events are not tools"));
}

#[test]
fn execution_section_hides_receiver_examples_without_resource_operations() {
    let surface = prompt_surface(
        lashlang::ResourceCatalog::new(),
        lashlang::LashlangAbilities::default(),
    );
    let section = rlm_execution_section_for_surface(RlmPromptFeatures::default(), &surface);

    assert!(!section.contains("TOOL.default"));
    assert!(!section.contains("Showcased Tools"));
    assert!(!section.contains("Resource operations"));
    assert!(section.contains("No receiver operations are available"));
}

#[test]
fn execution_section_does_not_advertise_unregistered_peer_capability() {
    let section =
        rlm_execution_section_for_surface(RlmPromptFeatures::default(), &full_prompt_surface());

    assert!(!section.contains("capability: \"peer\""));
    assert!(!section.contains("`peer`"));
}

#[test]
fn execution_section_keeps_tool_specific_examples_out_of_core_prompt() {
    let section =
        rlm_execution_section_for_surface(RlmPromptFeatures::default(), &full_prompt_surface());

    for tool_name in [
        "read_file",
        "exec_command",
        "apply_patch",
        "llm_query",
        "spawn_agent",
        "continue_as",
        "list_process_handles",
    ] {
        assert!(
            !section.contains(tool_name),
            "core RLM prompt should not mention tool-specific example `{tool_name}`"
        );
    }
}

#[test]
fn execution_section_can_disable_image_guidance() {
    let section = rlm_execution_section_for_surface(
        RlmPromptFeatures {
            images: false,
            ..RlmPromptFeatures::default()
        },
        &full_prompt_surface(),
    );

    assert!(!section.contains("Image"));
    assert!(!section.contains("image.size"));
    assert!(section.contains("### Language"));
    assert!(section.contains("### Builtins"));
    assert!(section.contains("### Common patterns"));
    assert!(section.contains("### Type literals"));
}

#[test]
fn execution_section_states_no_while_loop() {
    let section =
        rlm_execution_section_for_surface(RlmPromptFeatures::default(), &full_prompt_surface());

    assert!(section.contains("There is no `while` loop"));
    assert!(section.contains("use bounded `for` loops over ranges/lists"));
}

#[test]
fn fence_detector_accepts_inline_opener_after_prose() {
    // Regression: reasoning models emit the opening fence mid-line:
    // `…required output shape.```lashlang\n…`. Requiring newline
    // before ``` caused the detector to miss the block entirely,
    // which made the RLM turn terminate after one protocol_iteration
    // without executing anything.
    let text = "I'll inspect the prompt.```lashlang\nprint slice(input.prompt, 0, 10)\n```";
    let extraction = extract_first_lashlang_fence(text)
        .expect("inline opener with newline-terminated closer should parse");
    assert_eq!(extraction.code, "print slice(input.prompt, 0, 10)");
    assert!(contains_closed_lashlang_fence(text));
}

#[test]
fn fence_detector_still_accepts_newline_preceded_opener() {
    let text = "prose\n\n```lashlang\nsubmit 1\n```";
    let extraction = extract_first_lashlang_fence(text).expect("should parse");
    assert_eq!(extraction.code, "submit 1");
}

#[test]
fn fence_detector_closer_matches_anywhere() {
    // `\`\`\`` closes the block wherever it appears. Simpler mental
    // model: `\`\`\`lashlang` starts, `\`\`\`` stops. No newline
    // requirement on either side.
    let text = "```lashlang\nsubmit 1``` more prose";
    let extraction = extract_first_lashlang_fence(text).expect("should extract");
    assert_eq!(extraction.code, "submit 1");
}

#[test]
fn fence_detector_recovers_from_double_triple_concatenation() {
    // Reasoning-mode output sometimes emits ``` ``` back-to-back
    // (closer of one block immediately followed by opener of the
    // next with no prose between). The detector should still find
    // the first valid block.
    let text = "lead-in.```lashlang\nprint 1\n``````lashlang\nprint 2\n```";
    let extraction = extract_first_lashlang_fence(text)
        .expect("should extract the first block even with glued-on second block");
    assert_eq!(extraction.code, "print 1");
    assert!(extraction.had_extra_fences);
}

#[test]
fn fence_detector_ignores_unknown_lang_tag() {
    // `python` is not lashlang — the detector must look further.
    let text = "```python\nprint('x')\n```\n\n```lashlang\nsubmit 1\n```";
    let extraction = extract_first_lashlang_fence(text).expect("should skip python block");
    assert_eq!(extraction.code, "submit 1");
}

#[test]
fn fence_detector_four_backticks_allows_embedded_triple() {
    // Variable-length fences: opener of 4 backticks lets the body
    // contain literal ``` (which would otherwise terminate a 3-bt
    // block). Closer must be ≥4 backticks.
    let text = "````lashlang\nprint \"```\"\nsubmit 1\n````";
    let extraction = extract_first_lashlang_fence(text)
        .expect("4-backtick fence should allow embedded triple-backticks");
    assert_eq!(extraction.code, "print \"```\"\nsubmit 1");
    assert!(contains_closed_lashlang_fence(text));
}

#[test]
fn fence_detector_four_backtick_opener_accepts_longer_closer() {
    // CommonMark allows the closer to be longer than the opener.
    // 4-backtick opener closed by 5-backtick closer.
    let text = "````lashlang\nsubmit 1\n`````";
    let extraction =
        extract_first_lashlang_fence(text).expect("4-bt opener should accept 5-bt closer");
    assert_eq!(extraction.code, "submit 1");
}

#[test]
fn fence_detector_four_backtick_opener_ignores_inner_triple() {
    // Inner ``` runs (length 3 < opener 4) are NOT closers — body
    // continues until a run of ≥4 backticks (or EOF).
    let text = "````lashlang\nbody with ``` inside\nmore body\n````\ntail";
    let extraction = extract_first_lashlang_fence(text)
        .expect("4-bt opener should not be closed by inner triple");
    assert_eq!(extraction.code, "body with ``` inside\nmore body");
}
