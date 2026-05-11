use std::any::Any;
use std::sync::Arc;

use lash::{
    ModeSessionExtension, ModeTurnExtension, ModeTurnExtensionHandle, PromptContribution, TurnInput,
};

pub(crate) const RLM_TURN_INPUT_PLUGIN_ID: &str = "rlm";
use lashlang::{
    ProjectedBindingError, ProjectedBindings, ProjectedHostValue, ProjectedValue,
    Value as FlowValue,
};

#[derive(Clone, Default)]
pub struct RlmProjectedBindings {
    bindings: ProjectedBindings,
}

impl RlmProjectedBindings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bind_value(
        mut self,
        name: impl Into<String>,
        value: impl Into<FlowValue>,
    ) -> Result<Self, ProjectedBindingError> {
        let name = name.into();
        self.bindings.try_insert(
            name.clone(),
            ProjectedValue::scalar(name.clone(), value.into()),
        )?;
        Ok(self)
    }

    pub fn bind_json(
        self,
        name: impl Into<String>,
        value: serde_json::Value,
    ) -> Result<Self, ProjectedBindingError> {
        self.bind_value(name, lashlang::from_json(value))
    }

    pub fn bind_lazy(
        mut self,
        name: impl Into<String>,
        value: Arc<dyn ProjectedHostValue>,
    ) -> Result<Self, ProjectedBindingError> {
        let name = name.into();
        self.bindings
            .try_insert(name.clone(), ProjectedValue::custom(name.clone(), value))?;
        Ok(self)
    }

    pub fn names(&self) -> impl Iterator<Item = String> + '_ {
        self.bindings.names()
    }

    pub(crate) fn into_projected_bindings(self) -> ProjectedBindings {
        self.bindings
    }

    pub fn merge(mut self, other: Self) -> Result<Self, ProjectedBindingError> {
        for name in other.names() {
            let value = other
                .bindings
                .get(&name)
                .expect("name came from projected bindings");
            self.bindings.try_insert(name, value)?;
        }
        Ok(self)
    }

    /// Hydrate from a wire-format `RlmProjectedSeedSnapshot`. Each entry is
    /// re-projected via `bind_json`. Used by RLM mode to seed projections on a
    /// child session (spawn_agent / continue_as) from the parent's classified
    /// seed map.
    pub fn from_snapshot(
        snapshot: &lash_rlm_types::RlmProjectedSeedSnapshot,
    ) -> Result<Self, ProjectedBindingError> {
        let mut out = Self::new();
        for (name, value) in &snapshot.entries {
            out = out.bind_json(name.clone(), value.clone())?;
        }
        Ok(out)
    }
}

#[derive(Clone, Default)]
pub(crate) struct RlmProjectionExtension {
    pub(crate) bindings: RlmProjectedBindings,
}

impl RlmProjectionExtension {
    pub(crate) fn new(bindings: RlmProjectedBindings) -> Self {
        Self { bindings }
    }

    pub(crate) fn prompt_contributions_for(
        bindings: &RlmProjectedBindings,
    ) -> Vec<PromptContribution> {
        let mut names = bindings.names().collect::<Vec<_>>();
        if names.is_empty() {
            return Vec::new();
        }
        names.sort();
        let mut lines = vec![
            "These host-projected variables are already bound in lashlang. Access them directly in fenced `lashlang` code; do not recreate them manually.".to_string(),
            String::new(),
            "Host-projected variables:".to_string(),
        ];
        for name in names {
            lines.push(format!(
                "- `{name}`: projected host value, Readonly: true, projected binding"
            ));
        }
        vec![PromptContribution::environment(
            "Host Projected Variables",
            lines.join("\n"),
        )]
    }
}

impl ModeTurnExtension for RlmProjectionExtension {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn prompt_contributions(&self) -> Vec<PromptContribution> {
        Self::prompt_contributions_for(&self.bindings)
    }
}

impl ModeSessionExtension for RlmProjectionExtension {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub fn rlm_session_projection_extension(
    bindings: RlmProjectedBindings,
) -> lash::ModeSessionExtensionHandle {
    lash::ModeSessionExtensionHandle::new(RlmProjectionExtension::new(bindings))
}

pub trait RlmTurnInputExt {
    fn rlm_project(self, bindings: RlmProjectedBindings) -> Result<Self, ProjectedBindingError>
    where
        Self: Sized;
}

impl RlmTurnInputExt for TurnInput {
    fn rlm_project(
        mut self,
        bindings: RlmProjectedBindings,
    ) -> Result<Self, ProjectedBindingError> {
        let bindings = if let Some(existing) = self
            .turn_context
            .plugin_input::<RlmProjectionExtension>(RLM_TURN_INPUT_PLUGIN_ID)
            .cloned()
        {
            existing.bindings.clone().merge(bindings)?
        } else {
            bindings
        };
        self.turn_context.insert_plugin_input(
            RLM_TURN_INPUT_PLUGIN_ID,
            RlmProjectionExtension::new(bindings),
        );
        self.mode_extension = Some(ModeTurnExtensionHandle::new(RlmProjectionExtension::new(
            self.turn_context
                .plugin_input::<RlmProjectionExtension>(RLM_TURN_INPUT_PLUGIN_ID)
                .expect("RLM projection was just inserted")
                .bindings
                .clone(),
        )));
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bind_rejects_duplicate_names() {
        let duplicate = RlmProjectedBindings::new()
            .bind_json("current_query", serde_json::json!("first"))
            .expect("first bind")
            .bind_json("current_query", serde_json::json!("second"));
        let Err(err) = duplicate else {
            panic!("duplicate bind should fail");
        };
        assert_eq!(err.name(), "current_query");
    }

    #[test]
    fn merge_rejects_session_turn_duplicates() {
        let session = RlmProjectedBindings::new()
            .bind_json("current_query", serde_json::json!("session"))
            .expect("session bind");
        let turn = RlmProjectedBindings::new()
            .bind_json("current_query", serde_json::json!("turn"))
            .expect("turn bind");
        let duplicate = session.merge(turn);
        let Err(err) = duplicate else {
            panic!("duplicate session and turn binding should fail");
        };
        assert_eq!(err.name(), "current_query");
    }

    #[test]
    fn turn_input_extension_attaches_prompt_contribution() {
        let input = TurnInput {
            items: Vec::new(),
            image_blobs: Default::default(),
            mode: None,
            mode_turn_options: None,
            trace_turn_id: None,
            mode_extension: None,
            turn_context: lash::TurnContext::default(),
        }
        .rlm_project(
            RlmProjectedBindings::new()
                .bind_json("current_file", serde_json::json!("src/lib.rs"))
                .expect("bind"),
        )
        .expect("attach");
        let contribution = input
            .mode_extension
            .expect("extension")
            .prompt_contributions()
            .pop()
            .expect("prompt contribution");
        assert!(contribution.content.contains("`current_file`"));
        assert!(contribution.content.contains("Readonly: true"));
    }

    #[test]
    fn turn_input_extension_is_skipped_by_serde() {
        let input = TurnInput {
            items: Vec::new(),
            image_blobs: Default::default(),
            mode: None,
            mode_turn_options: None,
            trace_turn_id: Some("stable".to_string()),
            mode_extension: None,
            turn_context: lash::TurnContext::default(),
        }
        .rlm_project(
            RlmProjectedBindings::new()
                .bind_json("current_file", serde_json::json!("src/lib.rs"))
                .expect("bind"),
        )
        .expect("attach");

        let encoded = serde_json::to_string(&input).expect("serialize");
        assert!(!encoded.contains("mode_extension"));
        assert!(!encoded.contains("current_file"));
        let decoded: TurnInput = serde_json::from_str(&encoded).expect("deserialize");
        assert!(decoded.mode_extension.is_none());
        assert_eq!(decoded.trace_turn_id.as_deref(), Some("stable"));
    }

    #[test]
    fn matching_trace_turn_ids_do_not_share_projection_extensions() {
        let first = TurnInput {
            items: Vec::new(),
            image_blobs: Default::default(),
            mode: None,
            mode_turn_options: None,
            trace_turn_id: Some("same-trace".to_string()),
            mode_extension: None,
            turn_context: lash::TurnContext::default(),
        }
        .rlm_project(
            RlmProjectedBindings::new()
                .bind_json("first_name", serde_json::json!("first"))
                .expect("bind"),
        )
        .expect("attach first");
        let second = TurnInput {
            items: Vec::new(),
            image_blobs: Default::default(),
            mode: None,
            mode_turn_options: None,
            trace_turn_id: Some("same-trace".to_string()),
            mode_extension: None,
            turn_context: lash::TurnContext::default(),
        }
        .rlm_project(
            RlmProjectedBindings::new()
                .bind_json("second_name", serde_json::json!("second"))
                .expect("bind"),
        )
        .expect("attach second");

        let first_extension = first
            .mode_extension
            .as_ref()
            .and_then(|extension| extension.as_any().downcast_ref::<RlmProjectionExtension>())
            .expect("first extension");
        let second_extension = second
            .mode_extension
            .as_ref()
            .and_then(|extension| extension.as_any().downcast_ref::<RlmProjectionExtension>())
            .expect("second extension");
        assert_eq!(
            first_extension.bindings.names().collect::<Vec<_>>(),
            vec!["first_name".to_string()]
        );
        assert_eq!(
            second_extension.bindings.names().collect::<Vec<_>>(),
            vec!["second_name".to_string()]
        );
    }
}
