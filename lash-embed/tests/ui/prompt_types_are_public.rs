fn main() {
    let template = lash_embed::prompt::PromptTemplate::new(vec![
        lash_embed::prompt::PromptTemplateSection::untitled(vec![
            lash_embed::prompt::PromptTemplateEntry::builtin(
                lash_embed::prompt::PromptBuiltin::MainAgentIntro,
            ),
            lash_embed::prompt::PromptTemplateEntry::slot(lash_embed::prompt::PromptSlot::Guidance),
        ]),
    ]);
    let contribution =
        lash_embed::prompt::PromptContribution::guidance("Host", "Host guidance").with_priority(1);
    let layer = lash_embed::prompt::PromptLayer::with_template(template)
        .with_contribution(contribution)
        .with_cleared_slot(lash_embed::prompt::PromptSlot::Environment);

    let _ = lash_embed::LashCore::standard().prompt_layer(layer);
    let _ = lash_embed::prompt::default_prompt_template();
}
