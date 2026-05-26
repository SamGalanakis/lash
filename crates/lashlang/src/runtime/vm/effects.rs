use std::sync::Arc;

use super::super::host::{
    AbilityOp, AbilityResult, ProcessEvent, ProcessEventKind, ProcessSignal, ProcessSleep,
    ProcessSleepKind, ProcessStart, ResourceOperation,
};
use super::super::{
    ExecutionHost, RuntimeError, Value, error_value, is_process_handle_record,
    record_with_capacity, success, unwrap_tool_result,
};
use super::Vm;
use super::control::VmOutcome;

#[derive(Clone, Copy)]
pub(super) enum VmEffect {
    ResourceCall { operation: usize, argc: usize },
    ResourceCallUnwrap { operation: usize, argc: usize },
    StartProcess { process: usize, keys: usize },
    AwaitHandle,
    ProcessSleep(ProcessSleepKind),
    WaitSignal,
    SignalRun,
    AwaitHandleUnwrap,
    CancelHandle,
    Print,
    ProcessEvent(ProcessEventKind),
    Submit,
    Finish,
    Fail,
}

impl<H: ExecutionHost> Vm<'_, H> {
    pub(super) async fn resolve_effect(
        &mut self,
        effect: VmEffect,
    ) -> Result<Option<VmOutcome>, RuntimeError> {
        match effect {
            VmEffect::ResourceCall { operation, argc } => {
                let (receiver, args) = self.drain_receiver_call(argc)?;
                let result = match self
                    .host
                    .perform(AbilityOp::ResourceOperation(ResourceOperation {
                        receiver,
                        operation: self.chunk.names[operation].text.to_string(),
                        args,
                    }))
                    .await
                {
                    Ok(AbilityResult::Value(value)) => success(value),
                    Ok(AbilityResult::Unit) => {
                        error_value("resource operation returned no value".to_string())
                    }
                    Err(error) => error_value(error.to_string()),
                };
                self.stack.push(result);
            }
            VmEffect::ResourceCallUnwrap { operation, argc } => {
                let (receiver, args) = self.drain_receiver_call(argc)?;
                let value = self
                    .host
                    .perform(AbilityOp::ResourceOperation(ResourceOperation {
                        receiver,
                        operation: self.chunk.names[operation].text.to_string(),
                        args,
                    }))
                    .await
                    .and_then(|result| result.into_value("resource operation"))
                    .map_err(|error| RuntimeError::ValueError {
                        message: format!("`?` unwrapped failed resource operation: {error}"),
                    })?;
                self.stack.push(value);
            }
            VmEffect::StartProcess { process, keys } => {
                let args = self.drain_record_from_stack(keys)?;
                let value = self
                    .host
                    .perform(AbilityOp::StartProcess(ProcessStart {
                        module: self.chunk.module.clone(),
                        linked_module: self.chunk.linked_module.clone(),
                        process: self.chunk.names[process].text.to_string(),
                        args,
                    }))
                    .await
                    .and_then(|result| result.into_value("process start"))
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("process start failed: {err}"),
                    })?;
                self.stack.push(value);
            }
            VmEffect::AwaitHandle => {
                let handle = self.pop_stack()?;
                let result = self.await_value(handle).await;
                self.stack.push(result);
            }
            VmEffect::ProcessSleep(kind) => {
                let value = self.pop_stack()?;
                self.host
                    .perform(AbilityOp::ProcessSleep(ProcessSleep { kind, value }))
                    .await
                    .and_then(|result| result.into_value("process sleep"))
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("process sleep failed: {err}"),
                    })?;
                self.last_value = Some(Value::Null);
                self.stack.push(Value::Null);
            }
            VmEffect::WaitSignal => {
                let value = self
                    .host
                    .perform(AbilityOp::WaitSignal)
                    .await
                    .and_then(|result| result.into_value("wait signal"))
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("wait signal failed: {err}"),
                    })?;
                self.stack.push(value);
            }
            VmEffect::SignalRun => {
                let payload = self.pop_stack()?;
                let run = self.pop_stack()?;
                self.host
                    .perform(AbilityOp::SignalRun(ProcessSignal { run, payload }))
                    .await
                    .and_then(|result| result.into_value("signal run"))
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("signal run failed: {err}"),
                    })?;
                self.last_value = Some(Value::Null);
                self.stack.push(Value::Null);
            }
            VmEffect::AwaitHandleUnwrap => {
                let handle = self.pop_stack()?;
                let result = self.await_value_unwrap(handle).await?;
                self.stack.push(result);
            }
            VmEffect::CancelHandle => {
                let handle = self.pop_stack()?;
                let value = self
                    .host
                    .perform(AbilityOp::Cancel(handle))
                    .await
                    .and_then(|result| result.into_value("cancel"))
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("cancel failed: {err}"),
                    })?;
                self.last_value = Some(value.clone());
                self.stack.push(value);
            }
            VmEffect::ProcessEvent(kind) => {
                let value = self.pop_stack()?;
                self.host
                    .perform(AbilityOp::ProcessEvent(ProcessEvent {
                        kind,
                        value: value.clone(),
                    }))
                    .await
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("process event failed: {err}"),
                    })?;
                self.last_value = Some(value.clone());
                self.stack.push(value);
            }
            VmEffect::Print => {
                let value = self.pop_stack()?;
                let host_value = match &value {
                    Value::Projected(projected) => Value::String(projected.render().await.into()),
                    _ => value.clone(),
                };
                self.host
                    .perform(AbilityOp::Print(host_value))
                    .await
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("print failed: {err}"),
                    })?;
                self.last_value = Some(Value::Null);
                self.stack.push(Value::Null);
            }
            VmEffect::Submit => {
                let value = self.pop_stack()?;
                let value = self
                    .host
                    .perform(AbilityOp::Submit(value))
                    .await
                    .and_then(|result| result.into_value("submit"))
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("submit failed: {err}"),
                    })?;
                return Ok(Some(VmOutcome::Finished(value)));
            }
            VmEffect::Finish => {
                let value = self.pop_stack()?;
                let value = self
                    .host
                    .perform(AbilityOp::Finish(value))
                    .await
                    .and_then(|result| result.into_value("finish"))
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("finish failed: {err}"),
                    })?;
                return Ok(Some(VmOutcome::ProcessFinished(value)));
            }
            VmEffect::Fail => {
                let value = self.pop_stack()?;
                let value = self
                    .host
                    .perform(AbilityOp::Fail(value))
                    .await
                    .and_then(|result| result.into_value("fail"))
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("fail failed: {err}"),
                    })?;
                return Ok(Some(VmOutcome::ProcessFailed(value)));
            }
        }
        Ok(None)
    }

    fn await_value(
        &self,
        handle: Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Value> + Send + '_>> {
        Box::pin(async move {
            match handle {
                Value::List(handles) => {
                    let mut values = Vec::with_capacity(handles.len());
                    for handle in handles.iter().cloned() {
                        values.push(self.await_value(handle).await);
                    }
                    Value::List(values.into())
                }
                Value::Record(handles) if is_process_handle_record(&handles) => {
                    match self
                        .host
                        .perform(AbilityOp::Await(Value::Record(handles)))
                        .await
                    {
                        Ok(AbilityResult::Value(value)) => success(value),
                        Ok(AbilityResult::Unit) => {
                            error_value("await returned no value".to_string())
                        }
                        Err(error) => error_value(error.to_string()),
                    }
                }
                Value::Record(handles) => {
                    let mut record = record_with_capacity(handles.len());
                    for entry in handles.entries.iter() {
                        record.insert_symbolized(
                            entry.symbol,
                            entry.name.clone(),
                            self.await_value(entry.value.clone()).await,
                        );
                    }
                    Value::Record(Arc::new(record))
                }
                handle => match self.host.perform(AbilityOp::Await(handle)).await {
                    Ok(AbilityResult::Value(value)) => success(value),
                    Ok(AbilityResult::Unit) => error_value("await returned no value".to_string()),
                    Err(error) => error_value(error.to_string()),
                },
            }
        })
    }

    async fn await_value_unwrap(&self, handle: Value) -> Result<Value, RuntimeError> {
        match handle {
            Value::Record(handles) if is_process_handle_record(&handles) => self
                .host
                .perform(AbilityOp::Await(Value::Record(handles)))
                .await
                .and_then(|result| result.into_value("await"))
                .map_err(|error| RuntimeError::ValueError {
                    message: format!("`?` unwrapped failed tool result: {error}"),
                }),
            Value::List(_) | Value::Record(_) => unwrap_tool_result(self.await_value(handle).await),
            handle => self
                .host
                .perform(AbilityOp::Await(handle))
                .await
                .and_then(|result| result.into_value("await"))
                .map_err(|error| RuntimeError::ValueError {
                    message: format!("`?` unwrapped failed tool result: {error}"),
                }),
        }
    }
}
