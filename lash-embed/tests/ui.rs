#[test]
fn model_selection_requires_model_and_variant_together() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/model_selection_requires_variant.rs");
    t.pass("tests/ui/prompt_types_are_public.rs");
}
