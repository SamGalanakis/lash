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

fn prompt_surface_with_features(
    resources: lashlang::ResourceCatalog,
    abilities: lashlang::LashlangAbilities,
    language_features: lashlang::LashlangLanguageFeatures,
) -> lashlang::LashlangSurface {
    lashlang::LashlangSurface::new(resources, abilities).with_language_features(language_features)
}

fn tool_resources() -> lashlang::ResourceCatalog {
    let mut resources = lashlang::ResourceCatalog::new();
    resources.add_module_operation(
        ["web"],
        "Web",
        "search",
        "search_web",
        lashlang::TypeExpr::Any,
        lashlang::TypeExpr::Any,
    );
    resources.add_module_operation(
        ["web"],
        "Web",
        "fetch",
        "fetch_url",
        lashlang::TypeExpr::Any,
        lashlang::TypeExpr::Any,
    );
    resources.add_module_operation(
        ["files"],
        "Files",
        "read",
        "read_file",
        lashlang::TypeExpr::Any,
        lashlang::TypeExpr::Any,
    );
    resources
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
fn execution_section_hides_label_annotations_when_disabled() {
    let section =
        rlm_execution_section_for_surface(RlmPromptFeatures::default(), &full_prompt_surface());

    assert!(!section.contains("@label"));
}

#[test]
fn execution_section_documents_static_label_annotations_when_enabled() {
    let surface = prompt_surface_with_features(
        tool_resources(),
        lashlang::LashlangAbilities::all(),
        lashlang::LashlangLanguageFeatures::default().with_label_annotations(),
    );
    let section = rlm_execution_section_for_surface(RlmPromptFeatures::default(), &surface);

    assert!(section.contains("@label(title: \"Label\")"));
    assert!(section.contains("@label(title: \"Label\", description: \"Details\")"));
    assert!(section.contains("Execution labels"));
    assert!(section.contains("important Lashlang phases"));
    assert!(section.contains("At top level, label meaningful setup"));
    assert!(section.contains("string literals"));
    assert!(!section.contains("process-map"));
    assert!(!section.contains("visual process statement"));
    assert!(!section.contains("color:"));
}

#[test]
fn execution_section_hides_sleep_and_signals_independently() {
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
fn execution_section_documents_unwrapped_process_await_for_finished_values() {
    let surface = prompt_surface(
        tool_resources(),
        lashlang::LashlangAbilities::default().with_processes(),
    );
    let section = rlm_execution_section_for_surface(RlmPromptFeatures::default(), &surface);

    assert!(section.contains("`await handle` waits and returns a result wrapper"));
    assert!(section.contains("`result = (await handle)?`"));
    assert!(section.contains("then read `result.field`"));
    assert!(!section.contains("terminal result returned by `await handle`"));
}

#[test]
fn execution_section_shows_sleep_without_processes() {
    let surface = prompt_surface(
        tool_resources(),
        lashlang::LashlangAbilities::default().with_sleep(),
    );
    let section = rlm_execution_section_for_surface(RlmPromptFeatures::default(), &surface);

    assert!(section.contains("sleep for"));
    assert!(!section.contains("process name"));
}

#[test]
fn execution_section_hides_trigger_registry_language_when_disabled() {
    let surface = prompt_surface(
        tool_resources(),
        lashlang::LashlangAbilities::default()
            .with_processes()
            .with_sleep()
            .with_process_signals(),
    );
    let section = rlm_execution_section_for_surface(RlmPromptFeatures::default(), &surface);

    assert!(!section.contains("Trigger registry"));
}

#[test]
fn execution_section_hides_trigger_registry_language_without_processes() {
    let surface = prompt_surface(
        tool_resources(),
        lashlang::LashlangAbilities::default().with_triggers(),
    );
    let section = rlm_execution_section_for_surface(RlmPromptFeatures::default(), &surface);

    assert!(!section.contains("Trigger registry"));
    assert!(!section.contains("triggers.register"));
}

#[test]
fn execution_section_lists_typed_operations_constructors_and_trigger_sources() {
    let mut resources = lashlang::ResourceCatalog::new();
    lashlang::add_trigger_resource_operations(&mut resources);
    resources
        .add_trigger_source_constructor(
            ["timer", "Schedule"],
            lashlang::TypeExpr::Object(vec![
                lashlang::TypeField {
                    name: "expr".into(),
                    ty: lashlang::TypeExpr::Str,
                    optional: false,
                },
                lashlang::TypeField {
                    name: "tz".into(),
                    ty: lashlang::TypeExpr::Str,
                    optional: true,
                },
            ]),
            lashlang::NamedDataType::object(
                "timer.Tick",
                vec![lashlang::TypeField {
                    name: "fired_at".into(),
                    ty: lashlang::TypeExpr::Str,
                    optional: false,
                }],
            )
            .expect("valid timer tick type"),
        )
        .expect("valid timer trigger source");
    let surface = prompt_surface(
        resources,
        lashlang::LashlangAbilities::default()
            .with_processes()
            .with_triggers(),
    );

    let section = rlm_execution_section_for_surface(RlmPromptFeatures::default(), &surface);

    assert!(section.contains("### Host Surface"));
    assert!(section.contains("`await triggers.register("));
    assert!(section.contains("inputs: dict"));
    assert!(section.contains("name: str?"));
    assert!(section.contains("`type timer.Tick = { fired_at: str }`"));
    assert!(
        section.contains("`timer.Schedule({ expr: str, tz: str? }) -> TriggerSource<timer.Tick>`")
    );
    assert!(
        section.contains(
            "`timer.Schedule` can be passed to `triggers.register` and emits `timer.Tick`"
        )
    );
}

#[test]
fn execution_section_hides_module_examples_without_module_operations() {
    let surface = prompt_surface(
        lashlang::ResourceCatalog::new(),
        lashlang::LashlangAbilities::default(),
    );
    let section = rlm_execution_section_for_surface(RlmPromptFeatures::default(), &surface);

    assert!(!section.contains("await tools."));
    assert!(!section.contains("Showcased Tools"));
    assert!(!section.contains("Module operations"));
    assert!(section.contains("No module operations are available"));
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
fn execution_section_mentions_while_and_bounded_loop_guidance() {
    let section =
        rlm_execution_section_for_surface(RlmPromptFeatures::default(), &full_prompt_surface());

    assert!(section.contains("statement `if`/`for`/`while`"));
    assert!(section.contains("Prefer bounded `while` loops where possible"));
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
