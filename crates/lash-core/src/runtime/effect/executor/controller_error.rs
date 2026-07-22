use crate::PluginError;
use crate::runtime::RuntimeError;

use serde::{Deserialize, Serialize};

use super::RuntimeEffectKind;

#[derive(Clone, Debug, thiserror::Error, Serialize, Deserialize)]
#[error("{code}: {message}")]
pub struct RuntimeEffectControllerError {
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<crate::runtime::effect::RuntimeEffectReplayMismatchSummary>,
}

impl RuntimeEffectControllerError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            summary: None,
        }
    }

    pub fn with_summary(
        mut self,
        summary: crate::runtime::effect::RuntimeEffectReplayMismatchSummary,
    ) -> Self {
        self.summary = Some(summary);
        self
    }

    pub(in crate::runtime::effect) fn wrong_outcome(
        expected: RuntimeEffectKind,
        actual: RuntimeEffectKind,
    ) -> Self {
        Self::new(
            "runtime_effect_wrong_outcome",
            format!(
                "expected {} outcome, got {}",
                expected.as_str(),
                actual.as_str()
            ),
        )
    }

    pub(crate) fn into_runtime_error(self) -> RuntimeError {
        RuntimeError::new(self.code, self.message)
    }
}

impl From<RuntimeError> for RuntimeEffectControllerError {
    fn from(err: RuntimeError) -> Self {
        Self::new(err.code.as_str(), err.message)
    }
}

impl From<PluginError> for RuntimeEffectControllerError {
    fn from(err: PluginError) -> Self {
        Self::new("plugin", err.to_string())
    }
}

impl From<crate::StoreError> for RuntimeEffectControllerError {
    fn from(err: crate::StoreError) -> Self {
        Self::new("runtime_store", err.to_string())
    }
}
