fn main() {
    let template = lash_embed::PromptTemplate::new(vec![
        lash_embed::PromptTemplateSection::untitled(vec![
            lash_embed::PromptTemplateEntry::builtin(lash_embed::PromptBuiltin::MainAgentIntro),
            lash_embed::PromptTemplateEntry::slot(lash_embed::PromptSlot::Guidance),
        ]),
    ]);
    let contribution =
        lash_embed::PromptContribution::guidance("Host", "Host guidance").with_priority(1);
    let layer = lash_embed::PromptLayer::with_template(template)
        .with_contribution(contribution)
        .with_cleared_slot(lash_embed::PromptSlot::Environment);

    let _ = lash_embed::LashCore::standard().prompt_layer(layer);
    let _ = lash_embed::default_prompt_template();
}
