use std::any::Any;
use std::sync::Arc;

use lash::{ModeTurnSidecar, ModeTurnSidecarHandle, PromptContribution, TurnInput};
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
}

#[derive(Clone, Default)]
pub(crate) struct RlmProjectionSidecar {
    pub(crate) bindings: RlmProjectedBindings,
}

impl RlmProjectionSidecar {
    pub(crate) fn new(bindings: RlmProjectedBindings) -> Self {
        Self { bindings }
    }
}

impl ModeTurnSidecar for RlmProjectionSidecar {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn prompt_contributions(&self) -> Vec<PromptContribution> {
        let mut names = self.bindings.names().collect::<Vec<_>>();
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
        let bindings = if let Some(existing) = self.mode_sidecar_handle().and_then(|sidecar| {
            sidecar
                .as_any()
                .downcast_ref::<RlmProjectionSidecar>()
                .cloned()
        }) {
            existing.bindings.clone().merge(bindings)?
        } else {
            bindings
        };
        self.set_mode_sidecar(ModeTurnSidecarHandle::new(RlmProjectionSidecar::new(
            bindings,
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
            user_input: None,
            mode: None,
            mode_turn_options: None,
            trace_turn_id: None,
        }
        .rlm_project(
            RlmProjectedBindings::new()
                .bind_json("current_file", serde_json::json!("src/lib.rs"))
                .expect("bind"),
        )
        .expect("attach");
        let contribution = input
            .mode_sidecar_handle()
            .expect("sidecar")
            .prompt_contributions()
            .pop()
            .expect("prompt contribution");
        assert!(contribution.content.contains("`current_file`"));
        assert!(contribution.content.contains("Readonly: true"));
    }
}
