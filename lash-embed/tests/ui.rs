#[test]
fn model_selection_requires_model_and_variant_together() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/model_selection_requires_variant.rs");
    t.compile_fail("tests/ui/root_tool_provider_is_not_public.rs");
    t.compile_fail("tests/ui/session_control_flat_methods_are_not_public.rs");
    t.pass("tests/ui/prompt_types_are_public.rs");
}
