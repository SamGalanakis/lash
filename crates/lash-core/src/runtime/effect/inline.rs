use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use futures_util::FutureExt;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::PluginError;
use crate::runtime::host::{ProcessAwaitOutput, ProcessRecord};

use super::controller::{RuntimeEffectController, RuntimeEffectControllerError};
use super::envelope::{
    ProcessCommand, ProcessEffectOutcome, RuntimeEffectCommand, RuntimeEffectEnvelope,
    RuntimeEffectOutcome,
};
use super::local::{ProcessRunner, RuntimeEffectLocalExecutor};

/// Default in-process effect controller.
#[derive(Clone, Default)]
pub struct InlineRuntimeEffectController {
    processes: Arc<Mutex<HashMap<String, LocalProcessExecution>>>,
}

#[async_trait::async_trait]
impl RuntimeEffectController for InlineRuntimeEffectController {
    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        match envelope.command.clone() {
            RuntimeEffectCommand::Process { command } => {
                let result = self
                    .execute_process_command(command, local_executor)
                    .await?;
                Ok(RuntimeEffectOutcome::Process { result })
            }
            _ => local_executor.execute(envelope).await,
        }
    }
}

impl InlineRuntimeEffectController {
    pub(crate) async fn start_process(
        &self,
        registry: Arc<dyn crate::ProcessRegistry>,
        registration: crate::ProcessRegistration,
        grant: Option<crate::ProcessStartGrant>,
        execution_context: crate::ProcessExecutionContext,
        runner: Arc<dyn ProcessRunner>,
    ) -> Result<ProcessRecord, PluginError> {
        let record = registry.register_process(registration.clone()).await?;
        if let Some(grant) = grant {
            registry
                .grant_handle(&grant.session_id, &registration.id, grant.descriptor)
                .await?;
        }
        let process_id = registration.id.clone();
        let cancellation = CancellationToken::new();
        let task_cancellation = cancellation.clone();
        let registry_for_task = Arc::clone(&registry);
        let processes = Arc::clone(&self.processes);
        let process_id_for_task = process_id.clone();
        let handle = tokio::spawn(async move {
            let future = runner.run_process(
                registration,
                execution_context,
                Arc::clone(&registry_for_task),
                task_cancellation,
            );
            let output = AssertUnwindSafe(future).catch_unwind().await;
            let output = match output {
                Ok(output) => output,
                Err(_) => ProcessAwaitOutput::from_tool_output(crate::ToolCallOutput::failure(
                    crate::ToolFailure::runtime(
                        crate::ToolFailureClass::Internal,
                        "process_panicked",
                        "process panicked",
                    ),
                )),
            };
            let _ = registry_for_task
                .complete_process(&process_id_for_task, output)
                .await;
            processes.lock().await.remove(&process_id_for_task);
        });
        self.processes
            .lock()
            .await
            .insert(process_id, LocalProcessExecution { cancellation });
        drop(handle);
        Ok(record)
    }

    pub(crate) async fn request_process_cancel(
        &self,
        registry: Arc<dyn crate::ProcessRegistry>,
        process_id: &str,
        reason: Option<String>,
    ) -> Result<ProcessRecord, PluginError> {
        let _ = reason;
        let record = registry
            .get_process(process_id)
            .await
            .ok_or_else(|| PluginError::Session(format!("unknown process `{process_id}`")))?;
        let execution = self.processes.lock().await.get(process_id).cloned();
        if let Some(execution) = execution {
            execution.cancellation.cancel();
        }
        Ok(record)
    }

    async fn execute_process_command(
        &self,
        command: ProcessCommand,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<ProcessEffectOutcome, RuntimeEffectControllerError> {
        let execution = local_executor.into_process()?;
        let registry = execution.registry;
        match command {
            ProcessCommand::Start {
                registration,
                grant,
                execution_context,
            } => {
                let Some(runner) = execution.runner else {
                    return Err(RuntimeEffectControllerError::new(
                        "process_runner_required",
                        format!(
                            "process `{}` cannot be started without a runtime runner",
                            registration.id
                        ),
                    ));
                };
                let record = self
                    .start_process(registry, registration, grant, execution_context, runner)
                    .await?;
                Ok(ProcessEffectOutcome::Start { record })
            }
            ProcessCommand::List { session_id } => {
                let entries = registry.list_handle_grants(&session_id).await?;
                Ok(ProcessEffectOutcome::List { entries })
            }
            ProcessCommand::Transfer {
                from_session_id,
                to_session_id,
                process_ids,
            } => {
                registry
                    .transfer_handle_grants(&from_session_id, &to_session_id, &process_ids)
                    .await?;
                Ok(ProcessEffectOutcome::Transfer)
            }
            ProcessCommand::Await { process_id } => {
                let output = registry.await_process(&process_id).await?;
                Ok(ProcessEffectOutcome::Await { output })
            }
            ProcessCommand::Cancel { process_id, reason } => {
                let record = self
                    .request_process_cancel(registry, &process_id, reason)
                    .await?;
                Ok(ProcessEffectOutcome::Cancel { record })
            }
        }
    }
}

#[derive(Clone)]
struct LocalProcessExecution {
    cancellation: CancellationToken,
}

impl std::fmt::Debug for InlineRuntimeEffectController {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InlineRuntimeEffectController")
            .finish_non_exhaustive()
    }
}
