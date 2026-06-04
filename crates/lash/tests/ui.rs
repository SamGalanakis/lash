#[test]
fn model_selection_requires_model_and_variant_together() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/model_selection_requires_variant.rs");
    t.compile_fail("tests/ui/root_tool_provider_is_not_public.rs");
    t.compile_fail("tests/ui/session_control_flat_methods_are_not_public.rs");
    t.compile_fail("tests/ui/effect_host_activation_methods_are_removed.rs");
    t.pass("tests/ui/config_control_update_session_config_returns_result.rs");
    t.pass("tests/ui/durable_builder_without_advanced.rs");
    t.pass("tests/ui/facade_boundary_types_are_public.rs");
    t.pass("tests/ui/prompt_types_are_public.rs");
}
