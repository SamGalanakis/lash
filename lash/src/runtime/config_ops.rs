//! `LashRuntime` configuration mutators: provider, model, context-window,
//! session id, and tool-surface refresh.
//!
//! Extracted from `runtime/mod.rs`. This file re-opens `impl LashRuntime`;
//! no types live here and no public API is changed.

use crate::SessionError;
use crate::provider::ProviderHandle;

use super::LashRuntime;

impl LashRuntime {
    /// Update model on the runtime config.
    pub fn set_model(&mut self, model: String) {
        self.policy.model = model;
        self.state.policy.model = self.policy.model.clone();
    }

    /// Update model variant on the runtime config.
    pub fn set_model_variant(&mut self, model_variant: Option<String>) {
        self.policy.model_variant = model_variant;
        self.state.policy.model_variant = self.policy.model_variant.clone();
    }

    /// Update explicit model context metadata on the runtime config.
    pub fn set_max_context_tokens(&mut self, max_context_tokens: usize) {
        self.policy.max_context_tokens = Some(max_context_tokens);
        self.state.policy.max_context_tokens = self.policy.max_context_tokens;
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
        model: Option<String>,
        model_variant: Option<Option<String>>,
        max_context_tokens: Option<usize>,
        prompt: Option<crate::PromptLayer>,
    ) {
        let previous = self.session_policy();
        if let Some(provider) = provider {
            self.policy.provider = provider;
        }
        if let Some(model) = model {
            self.policy.model = model;
        }
        if let Some(model_variant) = model_variant {
            self.policy.model_variant = model_variant;
        }
        if let Some(max_context_tokens) = max_context_tokens {
            self.policy.max_context_tokens = Some(max_context_tokens);
        }
        if let Some(prompt) = prompt {
            self.policy.prompt = prompt;
        }
        self.state.policy = self.policy.clone();
        // Eagerly compact messages if the context window shrunk.
        let new_max = self.policy.max_context_tokens;
        let old_max = previous.max_context_tokens;
        if new_max < old_max || (new_max.is_some() && old_max.is_none()) {
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
        self.update_session_config(None, None, None, None, Some(prompt))
            .await;
    }

    pub async fn clear_prompt_template(&mut self) {
        let mut prompt = self.policy.prompt.clone();
        prompt.template = None;
        self.update_session_config(None, None, None, None, Some(prompt))
            .await;
    }

    pub async fn add_prompt_contribution(&mut self, contribution: crate::PromptContribution) {
        let mut prompt = self.policy.prompt.clone();
        prompt.add_contribution(contribution);
        self.update_session_config(None, None, None, None, Some(prompt))
            .await;
    }

    pub async fn replace_prompt_slot(
        &mut self,
        slot: crate::PromptSlot,
        contributions: impl IntoIterator<Item = crate::PromptContribution>,
    ) {
        let mut prompt = self.policy.prompt.clone();
        prompt.replace_slot(slot, contributions);
        self.update_session_config(None, None, None, None, Some(prompt))
            .await;
    }

    pub async fn clear_prompt_slot(&mut self, slot: crate::PromptSlot) {
        let mut prompt = self.policy.prompt.clone();
        prompt.clear_slot(slot);
        self.update_session_config(None, None, None, None, Some(prompt))
            .await;
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

    pub async fn apply_dynamic_tool_state(
        &mut self,
        snapshot: crate::DynamicStateSnapshot,
    ) -> Result<u64, SessionError> {
        let Some(session) = self.session.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        let Some(dynamic_tools) = session.plugins().dynamic_tools() else {
            return Err(SessionError::Protocol(
                "dynamic tools are unavailable in this runtime session".to_string(),
            ));
        };

        let generation = dynamic_tools.apply_state(snapshot).map_err(|err| {
            SessionError::Protocol(format!("dynamic tool reconfigure failed: {err}"))
        })?;
        self.stamp_live_plugin_state();
        Ok(generation)
    }
}
