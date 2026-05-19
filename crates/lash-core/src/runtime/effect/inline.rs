use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use futures_util::FutureExt;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::PluginError;
use crate::runtime::host::{
    BackgroundTaskCompletion, BackgroundTaskRecord, BackgroundTaskRegistration,
    BackgroundTaskRegistry, BackgroundTaskState,
};

use super::controller::{RuntimeEffectController, RuntimeEffectControllerError};
use super::envelope::{RuntimeEffectEnvelope, RuntimeEffectOutcome};
use super::local::{
    BackgroundTaskLocalExecutor, LocalBackgroundCancelPolicy, RuntimeEffectLocalExecutor,
};

/// Default in-process effect controller.
#[derive(Clone, Default)]
pub struct InlineRuntimeEffectController {
    background_tasks: Arc<Mutex<HashMap<String, LocalBackgroundExecution>>>,
}

#[async_trait::async_trait]
impl RuntimeEffectController for InlineRuntimeEffectController {
    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        local_executor.execute(envelope).await
    }

    async fn start_background_task(
        &self,
        registry: Arc<dyn BackgroundTaskRegistry>,
        registration: BackgroundTaskRegistration,
        local_executor: BackgroundTaskLocalExecutor,
    ) -> Result<BackgroundTaskRecord, PluginError> {
        let task_id = registration.id.clone();
        let running = registry.mark_running(&task_id).await?;
        let cancellation = CancellationToken::new();
        let task_cancellation = cancellation.clone();
        let registry_for_task = Arc::clone(&registry);
        let tasks = Arc::clone(&self.background_tasks);
        let task_id_for_task = task_id.clone();
        let cancel_policy = local_executor.cancel_policy();
        let handle = tokio::spawn(async move {
            let future = local_executor.run(task_cancellation);
            let completion = AssertUnwindSafe(future).catch_unwind().await;
            let completion = match completion {
                Ok(completion) if completion.state.is_terminal() => completion,
                Ok(completion) => BackgroundTaskCompletion {
                    state: BackgroundTaskState::Failed,
                    summary: Some(format!(
                        "background task returned non-terminal state {:?}",
                        completion.state
                    )),
                    output: completion.output,
                },
                Err(_) => BackgroundTaskCompletion {
                    state: BackgroundTaskState::Failed,
                    summary: Some("background task panicked".to_string()),
                    output: Some(crate::ToolCallOutput::failure(crate::ToolFailure::runtime(
                        crate::ToolFailureClass::Internal,
                        "background_task_panicked",
                        "background task panicked",
                    ))),
                },
            };
            let _ = registry_for_task
                .complete(&task_id_for_task, completion)
                .await;
            tasks.lock().await.remove(&task_id_for_task);
        });
        self.background_tasks.lock().await.insert(
            task_id,
            LocalBackgroundExecution {
                cancellation,
                abort: handle.abort_handle(),
                cancel_policy,
            },
        );
        drop(handle);
        Ok(running)
    }

    async fn request_background_task_cancel(
        &self,
        registry: Arc<dyn BackgroundTaskRegistry>,
        task_id: &str,
        reason: Option<String>,
    ) -> Result<BackgroundTaskRecord, PluginError> {
        let record = registry.request_cancel(task_id, reason.clone()).await?;
        let execution = self.background_tasks.lock().await.get(task_id).cloned();
        if let Some(execution) = execution {
            execution.cancellation.cancel();
            if matches!(
                execution.cancel_policy,
                LocalBackgroundCancelPolicy::LocalAbort
            ) {
                execution.abort.abort();
                self.background_tasks.lock().await.remove(task_id);
                let message = reason.unwrap_or_else(|| "background task was cancelled".to_string());
                return registry
                    .complete(
                        task_id,
                        BackgroundTaskCompletion {
                            state: BackgroundTaskState::Cancelled,
                            summary: Some(message.clone()),
                            output: Some(crate::ToolCallOutput::cancelled(
                                crate::ToolCancellation::runtime(message),
                            )),
                        },
                    )
                    .await;
            }
        }
        Ok(record)
    }
}

#[derive(Clone)]
struct LocalBackgroundExecution {
    cancellation: CancellationToken,
    abort: tokio::task::AbortHandle,
    cancel_policy: LocalBackgroundCancelPolicy,
}

impl std::fmt::Debug for InlineRuntimeEffectController {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InlineRuntimeEffectController")
            .finish_non_exhaustive()
    }
}
