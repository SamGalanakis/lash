use lash_core::{PromptContribution, PromptLayer, PromptSlot, PromptTemplate};

/// Builder-agnostic prompt-layer mutation surface.
///
/// Every builder that owns a [`PromptLayer`] — [`crate::LashCoreBuilder`] (the
/// runtime default prompt) and [`crate::SessionBuilder`] (the per-session
/// prompt) — implements this trait by exposing its layer via
/// [`prompt_layer_mut`](PromptLayerSink::prompt_layer_mut). The
/// template/contribution/slot operations are then defined here exactly once,
/// instead of being copy-pasted per builder.
///
/// (The per-turn `TurnBuilder` does not implement this trait: its prompt
/// operations forward to the turn context, which owns that logic, so there is
/// nothing to share.)
pub trait PromptLayerSink: Sized {
    /// Mutable access to the builder's prompt layer, created on first use.
    fn prompt_layer_mut(&mut self) -> &mut PromptLayer;

    /// Set the base prompt template.
    fn prompt_template(mut self, template: PromptTemplate) -> Self {
        self.prompt_layer_mut().template = Some(template);
        self
    }

    /// Add a single prompt contribution to its slot.
    fn prompt_contribution(mut self, contribution: PromptContribution) -> Self {
        self.prompt_layer_mut().add_contribution(contribution);
        self
    }

    /// Replace all contributions in a slot.
    fn replace_prompt_slot(
        mut self,
        slot: PromptSlot,
        contributions: impl IntoIterator<Item = PromptContribution>,
    ) -> Self {
        self.prompt_layer_mut().replace_slot(slot, contributions);
        self
    }

    /// Clear all contributions from a slot.
    fn clear_prompt_slot(mut self, slot: PromptSlot) -> Self {
        self.prompt_layer_mut().clear_slot(slot);
        self
    }

    /// Replace the whole prompt layer.
    fn prompt_layer(mut self, layer: PromptLayer) -> Self {
        *self.prompt_layer_mut() = layer;
        self
    }
}
