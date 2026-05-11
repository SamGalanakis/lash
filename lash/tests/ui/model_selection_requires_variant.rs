fn main() {
    let _ = lash::LashCore::standard().model("model-only");
    let _ = lash::LashCore::standard().model_variant("low");
}

fn turn_builder_model_requires_variant(builder: lash::TurnBuilder) {
    let _ = builder.model("model-only");
}
