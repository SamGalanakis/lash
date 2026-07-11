fn main() {
    let _ = lash::LashCore::standard_builder().model("model-only");
    let _ = lash::LashCore::standard_builder().model_variant("low");
}

fn turn_builder_has_no_model_overlay(builder: lash::TurnBuilder) {
    let _ = builder.model("model-only");
}
