use std::sync::Arc;

use crate::{LashlangExecutionCallSite, LashlangExecutionChild};

use super::super::host::{
    AbilityOp, AbilityResult, ProcessEvent, ProcessEventKind, ProcessSignal, ProcessStart,
    ResourceOperation, Sleep, SleepKind,
};
use super::super::{
    ExecutionHost, RuntimeError, Value, error_value, is_process_handle_record,
    record_with_capacity, success, unwrap_tool_result,
};
use super::control::VmOutcome;
use super::{ActiveLashlangExecutionNode, Vm};

#[derive(Clone, Copy)]
pub(super) enum VmEffect {
    ResourceCall { operation: usize, argc: usize },
    ResourceCallUnwrap { operation: usize, argc: usize },
    StartProcess { process: usize, keys: usize },
    AwaitHandle,
    Sleep(SleepKind),
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
        instruction_ip: usize,
    ) -> Result<Option<VmOutcome>, RuntimeError> {
        let active = self.begin_lashlang_execution(instruction_ip);
        let result = self.resolve_effect_inner(effect, active.as_ref()).await;
        match (&result, active.as_ref()) {
            (Ok(Some(VmOutcome::ProcessFailed(value))), Some(active)) => {
                self.fail_lashlang_execution(active, value.to_string());
            }
            (Ok(_), Some(active)) => {
                self.complete_lashlang_execution(active);
            }
            (Err(error), Some(active)) => {
                self.fail_lashlang_execution(active, error.to_string());
            }
            _ => {}
        }
        result
    }

    async fn resolve_effect_inner(
        &mut self,
        effect: VmEffect,
        active: Option<&ActiveLashlangExecutionNode>,
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
                        call_site: active.map(lashlang_execution_call_site),
                    }))
                    .await
                {
                    Ok(AbilityResult::Value(value)) => success(value),
                    Ok(AbilityResult::Unit) => {
                        error_value("module operation returned no value".to_string())
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
                        call_site: active.map(lashlang_execution_call_site),
                    }))
                    .await
                    .and_then(|result| result.into_value("module operation"))
                    .map_err(|error| RuntimeError::ValueError {
                        message: format!("`?` unwrapped failed module operation: {error}"),
                    })?;
                self.stack.push(value);
            }
            VmEffect::StartProcess { process, keys } => {
                let args = self.drain_record_from_stack(keys)?;
                let process_name = self.chunk.names[process].text.to_string();
                let module_context =
                    self.chunk
                        .module_context
                        .as_ref()
                        .ok_or_else(|| RuntimeError::ValueError {
                            message: "`start` requires a linked lashlang module artifact"
                                .to_string(),
                        })?;
                let process_ref = module_context
                    .process_refs
                    .get(&process_name)
                    .cloned()
                    .ok_or_else(|| RuntimeError::ValueError {
                        message: format!(
                            "linked lashlang module `{}` does not export process `{process_name}`",
                            module_context.module_ref
                        ),
                    })?;
                let child_module_ref = module_context.module_ref.clone();
                let child_required_surface_ref = module_context.required_surface_ref.clone();
                let value = self
                    .host
                    .perform(AbilityOp::StartProcess(Box::new(ProcessStart {
                        module_ref: child_module_ref.clone(),
                        process_ref: process_ref.clone(),
                        required_surface_ref: child_required_surface_ref,
                        process_name: process_name.clone(),
                        args,
                    })))
                    .await
                    .and_then(|result| result.into_value("process start"))
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("process start failed: {err}"),
                    })?;
                if let (Some(active), Some(process_id)) =
                    (active, process_handle_id_from_value(&value))
                {
                    self.observe_child_started(
                        active,
                        LashlangExecutionChild {
                            process_id,
                            module_ref: child_module_ref,
                            process_ref,
                            process_name,
                        },
                    );
                }
                self.stack.push(value);
            }
            VmEffect::AwaitHandle => {
                let handle = self.pop_stack()?;
                let result = self.await_value(handle).await;
                self.stack.push(result);
            }
            VmEffect::Sleep(kind) => {
                let value = self.pop_stack()?;
                self.host
                    .perform(AbilityOp::Sleep(Sleep { kind, value }))
                    .await
                    .and_then(|result| result.into_value("sleep"))
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("sleep failed: {err}"),
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

fn lashlang_execution_call_site(active: &ActiveLashlangExecutionNode) -> LashlangExecutionCallSite {
    LashlangExecutionCallSite {
        site: active.site.clone(),
        occurrence: active.occurrence,
    }
}

fn process_handle_id_from_value(value: &Value) -> Option<String> {
    let record = value.as_record()?;
    let Value::String(kind) = record.get("__handle__")? else {
        return None;
    };
    if kind.as_str() != "process" {
        return None;
    }
    let Value::String(id) = record.get("id")? else {
        return None;
    };
    Some(id.to_string())
}
