//! `LashRuntime` configuration mutators: provider, model spec, session id,
//! and tool-surface refresh.
//!
//! Extracted from `runtime/mod.rs`. This file re-opens `impl LashRuntime`;
//! no types live here and no public API is changed.

use crate::SessionError;
use crate::provider::ProviderHandle;

use super::LashRuntime;

impl LashRuntime {
    /// Update model spec on the runtime config.
    pub fn set_model(&mut self, model: crate::ModelSpec) {
        self.policy.model = model;
        self.state.policy.model = self.policy.model.clone();
    }

    /// Update provider on the runtime config.
    pub fn set_provider(&mut self, provider: ProviderHandle) {
        self.policy.provider = provider;
        self.state.policy.provider = self.policy.provider.clone();
    }

    /// Update session ID metadata on the runtime config.
    pub fn set_session_id(&mut self, session_id: Option<String>) {
        self.policy.session_id = session_id;
        self.state.policy.session_id = self.policy.session_id.clone();
    }

    pub async fn update_session_config(
        &mut self,
        provider: Option<ProviderHandle>,
        model: Option<crate::ModelSpec>,
        prompt: Option<crate::PromptLayer>,
    ) {
        let previous = self.session_policy();
        if let Some(provider) = provider {
            self.policy.provider = provider;
        }
        if let Some(model) = model {
            self.policy.model = model;
        }
        if let Some(prompt) = prompt {
            self.policy.prompt = prompt;
        }
        self.state.policy = self.policy.clone();
        // Eagerly compact messages if the context window shrunk.
        let new_max = Some(self.policy.context_window_tokens());
        let old_max = Some(previous.context_window_tokens());
        if new_max < old_max {
            let _ = self
                .rewrite_history(crate::RewriteTrigger::WindowShrink { old_max, new_max })
                .await;
        }
        self.apply_session_config_mutations(previous.clone()).await;
        self.notify_session_config_changed(previous).await;
    }

    pub async fn set_prompt_template(&mut self, template: crate::PromptTemplate) {
        let mut prompt = self.policy.prompt.clone();
        prompt.template = Some(template);
        self.update_session_config(None, None, Some(prompt)).await;
    }

    pub async fn clear_prompt_template(&mut self) {
        let mut prompt = self.policy.prompt.clone();
        prompt.template = None;
        self.update_session_config(None, None, Some(prompt)).await;
    }

    pub async fn add_prompt_contribution(&mut self, contribution: crate::PromptContribution) {
        let mut prompt = self.policy.prompt.clone();
        prompt.add_contribution(contribution);
        self.update_session_config(None, None, Some(prompt)).await;
    }

    pub async fn replace_prompt_slot(
        &mut self,
        slot: crate::PromptSlot,
        contributions: impl IntoIterator<Item = crate::PromptContribution>,
    ) {
        let mut prompt = self.policy.prompt.clone();
        prompt.replace_slot(slot, contributions);
        self.update_session_config(None, None, Some(prompt)).await;
    }

    pub async fn clear_prompt_slot(&mut self, slot: crate::PromptSlot) {
        let mut prompt = self.policy.prompt.clone();
        prompt.clear_slot(slot);
        self.update_session_config(None, None, Some(prompt)).await;
    }

    /// Re-register the current tool surface in the live RLM session.
    pub async fn refresh_session_tool_surface(&mut self) -> Result<(), SessionError> {
        let Some(session) = self.session.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        session.refresh_tool_surface().await?;
        self.stamp_live_plugin_state();
        Ok(())
    }

    pub async fn apply_tool_state(
        &mut self,
        snapshot: crate::ToolState,
    ) -> Result<u64, SessionError> {
        let Some(session) = self.session.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        let generation = session
            .plugins()
            .tool_registry()
            .apply_state(snapshot)
            .map_err(|err| SessionError::Protocol(format!("tool reconfigure failed: {err}")))?;
        self.stamp_live_plugin_state();
        Ok(generation)
    }
}
