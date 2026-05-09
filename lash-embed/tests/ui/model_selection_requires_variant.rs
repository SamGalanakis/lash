fn main() {
    let _ = lash_embed::LashCore::standard().model("model-only");
    let _ = lash_embed::LashCore::standard().model_variant("low");
}

fn turn_builder_model_requires_variant(builder: lash_embed::TurnBuilder<'_>) {
    let _ = builder.model("model-only");
}
