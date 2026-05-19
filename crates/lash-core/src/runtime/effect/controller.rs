use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{PluginError, RuntimeError};

use crate::runtime::host::{
    BackgroundTaskRecord, BackgroundTaskRegistration, BackgroundTaskRegistry,
};

use super::envelope::{RuntimeEffectEnvelope, RuntimeEffectKind, RuntimeEffectOutcome};
use super::local::{BackgroundTaskLocalExecutor, RuntimeEffectLocalExecutor};

/// Boundary for nondeterministic runtime work.
#[async_trait::async_trait]
pub trait RuntimeEffectController: Send + Sync {
    fn requires_durable_attachment_store(&self) -> bool {
        false
    }

    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError>;

    async fn start_background_task(
        &self,
        registry: Arc<dyn BackgroundTaskRegistry>,
        registration: BackgroundTaskRegistration,
        local_executor: BackgroundTaskLocalExecutor,
    ) -> Result<BackgroundTaskRecord, PluginError>;

    async fn request_background_task_cancel(
        &self,
        registry: Arc<dyn BackgroundTaskRegistry>,
        task_id: &str,
        reason: Option<String>,
    ) -> Result<BackgroundTaskRecord, PluginError>;
}

/// Borrowed durable effect controller for one runtime execution.
///
/// Durable integrations create one scope per externally
/// identified run and pass it to the scoped runtime entrypoints, making the
/// turn identity part of the idempotency contract rather than a tracing-only
/// hint.
#[derive(Clone, Copy)]
pub struct RuntimeEffectControllerScope<'run> {
    controller: &'run dyn RuntimeEffectController,
    turn_id: &'run str,
}

impl<'run> RuntimeEffectControllerScope<'run> {
    pub fn new(
        controller: &'run dyn RuntimeEffectController,
        turn_id: &'run str,
    ) -> Result<Self, RuntimeError> {
        if turn_id.is_empty() {
            return Err(RuntimeError {
                code: "missing_effect_scope_turn_id".to_string(),
                message: "scoped durable runs require a non-empty stable turn_id".to_string(),
            });
        }
        Ok(Self {
            controller,
            turn_id,
        })
    }

    pub fn controller(&self) -> &'run dyn RuntimeEffectController {
        self.controller
    }

    pub fn turn_id(&self) -> &'run str {
        self.turn_id
    }
}

/// Runtime-internal handle for effect-controller references carried through
/// per-turn execution contexts.
#[derive(Clone)]
pub(crate) enum RuntimeEffectControllerHandle<'run> {
    Borrowed(&'run dyn RuntimeEffectController),
    Shared(Arc<dyn RuntimeEffectController>),
}

impl<'run> RuntimeEffectControllerHandle<'run> {
    pub(crate) fn borrowed(controller: &'run dyn RuntimeEffectController) -> Self {
        Self::Borrowed(controller)
    }

    pub(crate) fn shared(controller: Arc<dyn RuntimeEffectController>) -> Self {
        Self::Shared(controller)
    }

    pub(crate) fn as_controller(&self) -> &dyn RuntimeEffectController {
        match self {
            Self::Borrowed(controller) => *controller,
            Self::Shared(controller) => controller.as_ref(),
        }
    }

    pub(crate) fn clone_scoped(&self) -> RuntimeEffectControllerHandle<'run> {
        self.clone()
    }
}

#[derive(Clone, Debug, thiserror::Error, Serialize, Deserialize)]
#[error("{code}: {message}")]
pub struct RuntimeEffectControllerError {
    pub code: String,
    pub message: String,
}

impl RuntimeEffectControllerError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }

    pub(super) fn wrong_outcome(expected: RuntimeEffectKind, actual: RuntimeEffectKind) -> Self {
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
        RuntimeError {
            code: self.code,
            message: self.message,
        }
    }
}

impl From<RuntimeError> for RuntimeEffectControllerError {
    fn from(err: RuntimeError) -> Self {
        Self::new(err.code, err.message)
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
