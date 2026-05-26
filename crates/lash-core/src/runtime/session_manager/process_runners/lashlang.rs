use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

impl RuntimeSessionManager {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime::session_manager::process_runners) async fn run_lashlang_process(
        &self,
        registration: crate::ProcessRegistration,
        registry: Arc<dyn crate::ProcessRegistry>,
        linked_module: ::lashlang::LinkedModule,
        process_name: String,
        args: serde_json::Map<String, serde_json::Value>,
        execution_context: crate::ProcessExecutionContext,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::ProcessAwaitOutput {
        let compiled = match ::lashlang::compile_linked_process(&linked_module, &process_name) {
            Ok(compiled) => compiled,
            Err(err) => {
                return process_lashlang_failure(
                    "process_compile_failed",
                    format!("failed to compile process `{process_name}`: {err}"),
                    None,
                );
            }
        };
        let mut globals = ::lashlang::Record::with_capacity(args.len());
        for (name, value) in args {
            globals.insert(name, ::lashlang::from_json(value));
        }
        let mut state = ::lashlang::State::from_snapshot(::lashlang::Snapshot { globals });

        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<crate::SessionEvent>(64);
        let event_drain = tokio::spawn(async move { while event_rx.recv().await.is_some() {} });
        let runtime_host = Arc::new(self.clone()) as Arc<dyn crate::plugin::RuntimeSessionHost>;
        let direct_completions = crate::DirectCompletionClient::runtime(
            Arc::new(self.clone()),
            crate::runtime::RuntimeEffectControllerHandle::shared(Arc::clone(
                &self.current.host.core.effect_controller,
            )),
            execution_context
                .tool_effect_metadata
                .as_ref()
                .and_then(|metadata| metadata.turn_id.clone()),
            self.current.turn_lease.clone(),
        );
        let dispatch = match self.current.plugins.tool_surface(
            &self.current.session_id,
            self.current.policy.execution_mode.clone(),
        ) {
            Ok(surface) => crate::tool_dispatch::ToolDispatchContext {
                plugins: Arc::clone(&self.current.plugins),
                tools: self.current.plugins.tools(),
                surface,
                host: Arc::clone(&runtime_host),
                processes: Arc::new(self.clone()),
                effect_controller: crate::runtime::RuntimeEffectControllerHandle::shared(
                    Arc::clone(&self.current.host.core.effect_controller),
                ),
                direct_completions: direct_completions.clone(),
                tool_effect_metadata: None,
                session_id: self.current.session_id.clone(),
                event_tx,
                turn_injection_bridge: crate::TurnInjectionBridge::new(),
                attachment_store: Arc::clone(&self.current.host.core.attachment_store),
                turn_context: crate::TurnContext::default(),
            },
            Err(err) => {
                drop(runtime_host);
                let _ = event_drain.await;
                return process_lashlang_failure(
                    "process_tool_surface_failed",
                    err.to_string(),
                    None,
                );
            }
        };
        let mut ctx = crate::ModeExecutionContext::new(
            self.current.session_id.clone(),
            self.current.policy.execution_mode.clone(),
            Arc::new(dispatch),
            self.current.plugins.lashlang_abilities(),
            Arc::clone(&self.current.host.core.attachment_store),
            Arc::new(crate::ChronologicalProjection::default()),
            None,
            crate::TurnContext::default(),
        )
        .with_cancellation_token(cancellation.clone());
        if let Some(metadata) = execution_context.tool_effect_metadata.clone() {
            ctx = ctx.with_effect_metadata(metadata);
        }

        let host = LashlangProcessHost {
            ctx,
            registry: Arc::clone(&registry),
            process_id: registration.id.clone(),
            wake_target_scope_key: execution_context.wake_target_scope_key,
            cancellation: cancellation.clone(),
            sleep_sequence: AtomicU64::new(0),
            signal_sequence: tokio::sync::Mutex::new(0),
        };
        let env = ::lashlang::ExecutionEnvironment::new(&host).process();

        let output = {
            tokio::select! {
                _ = cancellation.cancelled() => process_lashlang_cancelled("lashlang process was cancelled"),
                result = ::lashlang::execute(&compiled, &mut state, &env) => {
                    process_lashlang_execution_result(result)
                }
            }
        };
        drop(env);
        drop(host);
        let _ = event_drain.await;
        output
    }
}

struct LashlangProcessHost<'run> {
    ctx: crate::ModeExecutionContext<'run>,
    registry: Arc<dyn crate::ProcessRegistry>,
    process_id: String,
    wake_target_scope_key: Option<String>,
    cancellation: tokio_util::sync::CancellationToken,
    sleep_sequence: AtomicU64,
    signal_sequence: tokio::sync::Mutex<u64>,
}

impl LashlangProcessHost<'_> {
    fn resource_payload(
        &self,
        _receiver: &::lashlang::Value,
        args: &[::lashlang::Value],
    ) -> Result<serde_json::Value, ::lashlang::ExecutionHostError> {
        let mut payload = if let [::lashlang::Value::Record(record)] = args {
            crate::lashlang_bridge::lashlang_value_to_json(&::lashlang::Value::Record(Arc::clone(
                record,
            )))?
        } else {
            serde_json::json!({
                "args": args
                    .iter()
                    .map(crate::lashlang_bridge::lashlang_value_to_json)
                    .collect::<Result<Vec<_>, _>>()?,
            })
        };
        payload.as_object_mut().ok_or_else(|| {
            ::lashlang::ExecutionHostError::new("resource operation payload must be an object")
        })?;
        Ok(payload)
    }
}

impl LashlangProcessHost<'_> {
    async fn resource_operation(
        &self,
        operation: String,
        receiver: ::lashlang::Value,
        args: Vec<::lashlang::Value>,
    ) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        let resource = match &receiver {
            ::lashlang::Value::Resource(resource) => resource,
            _ => {
                return Err(::lashlang::ExecutionHostError::new(format!(
                    "resource operation `{operation}` requires a resource receiver"
                )));
            }
        };
        if resource.resource_type != "TOOL" || resource.alias != "default" {
            return Err(::lashlang::ExecutionHostError::new(format!(
                "resource `{}`.`{}` is not executable in this host",
                resource.resource_type, resource.alias
            )));
        }
        let manifest = self.ctx.callable_tool_manifest(&operation).ok_or_else(|| {
            ::lashlang::ExecutionHostError::new(format!(
                "resource operation `{operation}` is unavailable in this session"
            ))
        })?;
        let reply = self
            .ctx
            .call_tool(
                uuid::Uuid::new_v4().to_string(),
                manifest.name.clone(),
                self.resource_payload(&receiver, &args)?,
                0,
            )
            .await;
        mode_reply_to_lashlang_value(reply)
    }

    async fn await_handle(
        &self,
        handle: ::lashlang::Value,
    ) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        let reply = self
            .ctx
            .await_tool_handle(
                uuid::Uuid::new_v4().to_string(),
                crate::lashlang_bridge::lashlang_value_to_json(&handle)?,
            )
            .await;
        mode_reply_to_lashlang_value(reply)
    }

    async fn cancel_handle(
        &self,
        handle: ::lashlang::Value,
    ) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        let reply = self
            .ctx
            .cancel_tool_handle(
                uuid::Uuid::new_v4().to_string(),
                crate::lashlang_bridge::lashlang_value_to_json(&handle)?,
            )
            .await;
        mode_reply_to_lashlang_value(reply)
    }

    async fn start_process(
        &self,
        start: ::lashlang::ProcessStart,
    ) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        if start.module.process(&start.process).is_none()
            && let Some(manifest) = self.ctx.callable_tool_manifest(&start.process)
        {
            let payload = crate::lashlang_bridge::lashlang_value_to_json(
                &::lashlang::Value::Record(Arc::new(start.args)),
            )?;
            let reply = self
                .ctx
                .start_tool_call(
                    uuid::Uuid::new_v4().to_string(),
                    manifest.name.clone(),
                    payload,
                )
                .await;
            return mode_reply_to_lashlang_value(reply);
        }
        let (registration, label) = self
            .ctx
            .prepare_lashlang_process_start(start)
            .map_err(::lashlang::ExecutionHostError::new)?;
        let reply = self.ctx.start_lashlang_process(registration, label).await;
        mode_reply_to_lashlang_value(reply)
    }

    async fn process_event(
        &self,
        event: ::lashlang::ProcessEvent,
    ) -> Result<(), ::lashlang::ExecutionHostError> {
        let event_type = match event.kind {
            ::lashlang::ProcessEventKind::Yield => "process.yield",
            ::lashlang::ProcessEventKind::Wake => "process.wake",
        };
        self.registry
            .append_event(
                &self.process_id,
                crate::ProcessEventAppendRequest::new(
                    event_type,
                    crate::lashlang_bridge::process_event_payload(&event.value)?,
                )
                .with_optional_wake_target_scope_key(self.wake_target_scope_key.clone()),
            )
            .await
            .map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?;
        Ok(())
    }

    async fn process_sleep(
        &self,
        sleep: ::lashlang::ProcessSleep,
    ) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        let duration_ms = sleep_duration_ms(sleep.kind, &sleep.value)?;
        let sequence = self.sleep_sequence.fetch_add(1, Ordering::Relaxed);
        self.ctx
            .sleep_lashlang_process(&self.process_id, sequence, duration_ms)
            .await
            .map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?;
        Ok(::lashlang::Value::Null)
    }

    async fn wait_signal(&self) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        let after_sequence = *self.signal_sequence.lock().await;
        let wait =
            self.registry
                .wait_event_after(&self.process_id, "process.signal", after_sequence);
        let event = tokio::select! {
            _ = self.cancellation.cancelled() => {
                return Err(::lashlang::ExecutionHostError::new("wait signal was cancelled"));
            }
            event = wait => event.map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?,
        };
        *self.signal_sequence.lock().await = event.sequence;
        Ok(::lashlang::from_json(
            event
                .payload
                .get("payload")
                .cloned()
                .unwrap_or(event.payload),
        ))
    }

    async fn signal_run(
        &self,
        signal: ::lashlang::ProcessSignal,
    ) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        let target = process_id_from_lashlang_handle(&signal.run)?;
        let payload = crate::lashlang_bridge::lashlang_value_to_json(&signal.payload)?;
        self.registry
            .append_event(
                &target,
                crate::ProcessEventAppendRequest::new(
                    "process.signal",
                    serde_json::json!({
                        "payload": payload,
                        "timestamp": chrono::Utc::now().to_rfc3339(),
                    }),
                ),
            )
            .await
            .map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?;
        Ok(::lashlang::Value::Null)
    }
}

impl ::lashlang::ExecutionHost for LashlangProcessHost<'_> {
    async fn perform(
        &self,
        op: ::lashlang::AbilityOp,
    ) -> Result<::lashlang::AbilityResult, ::lashlang::ExecutionHostError> {
        match op {
            ::lashlang::AbilityOp::ResourceOperation(operation) => self
                .resource_operation(operation.operation, operation.receiver, operation.args)
                .await
                .map(::lashlang::AbilityResult::Value),
            ::lashlang::AbilityOp::Await(handle) => self
                .await_handle(handle)
                .await
                .map(::lashlang::AbilityResult::Value),
            ::lashlang::AbilityOp::Cancel(handle) => self
                .cancel_handle(handle)
                .await
                .map(::lashlang::AbilityResult::Value),
            ::lashlang::AbilityOp::StartProcess(start) => self
                .start_process(start)
                .await
                .map(::lashlang::AbilityResult::Value),
            ::lashlang::AbilityOp::ProcessEvent(event) => {
                self.process_event(event).await?;
                Ok(::lashlang::AbilityResult::Unit)
            }
            ::lashlang::AbilityOp::ProcessSleep(sleep) => self
                .process_sleep(sleep)
                .await
                .map(::lashlang::AbilityResult::Value),
            ::lashlang::AbilityOp::WaitSignal => self
                .wait_signal()
                .await
                .map(::lashlang::AbilityResult::Value),
            ::lashlang::AbilityOp::SignalRun(signal) => self
                .signal_run(signal)
                .await
                .map(::lashlang::AbilityResult::Value),
            ::lashlang::AbilityOp::Print(_) => Err(::lashlang::ExecutionHostError::new(
                "`print` is not available inside lashlang process bodies",
            )),
            ::lashlang::AbilityOp::Submit(value)
            | ::lashlang::AbilityOp::Finish(value)
            | ::lashlang::AbilityOp::Fail(value) => Ok(::lashlang::AbilityResult::Value(value)),
        }
    }

    async fn yield_now(&self) {
        tokio::task::yield_now().await;
    }
}

fn mode_reply_to_lashlang_value(
    reply: crate::ModeToolReply,
) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
    crate::lashlang_bridge::mode_tool_reply_to_lashlang_value(reply)
}

fn sleep_duration_ms(
    kind: ::lashlang::ProcessSleepKind,
    value: &::lashlang::Value,
) -> Result<u64, ::lashlang::ExecutionHostError> {
    match kind {
        ::lashlang::ProcessSleepKind::For => duration_value_ms(value),
        ::lashlang::ProcessSleepKind::Until => {
            let target = deadline_value_ms(value)?;
            let now = chrono::Utc::now().timestamp_millis();
            Ok(target.saturating_sub(now.max(0) as u64))
        }
    }
}

fn duration_value_ms(value: &::lashlang::Value) -> Result<u64, ::lashlang::ExecutionHostError> {
    match value {
        ::lashlang::Value::Number(value) if value.is_finite() && *value >= 0.0 => {
            Ok(value.round() as u64)
        }
        ::lashlang::Value::String(value) => parse_duration_ms(value),
        other => Err(::lashlang::ExecutionHostError::new(format!(
            "`sleep for` expects a non-negative millisecond number or duration string, got {other}"
        ))),
    }
}

fn deadline_value_ms(value: &::lashlang::Value) -> Result<u64, ::lashlang::ExecutionHostError> {
    match value {
        ::lashlang::Value::Number(value) if value.is_finite() && *value >= 0.0 => {
            Ok(value.round() as u64)
        }
        ::lashlang::Value::String(value) => chrono::DateTime::parse_from_rfc3339(value)
            .map(|deadline| deadline.timestamp_millis().max(0) as u64)
            .map_err(|err| {
                ::lashlang::ExecutionHostError::new(format!(
                    "`sleep until` expects RFC3339 text or Unix epoch milliseconds: {err}"
                ))
            }),
        other => Err(::lashlang::ExecutionHostError::new(format!(
            "`sleep until` expects RFC3339 text or Unix epoch milliseconds, got {other}"
        ))),
    }
}

fn parse_duration_ms(value: &str) -> Result<u64, ::lashlang::ExecutionHostError> {
    let value = value.trim();
    let (number, multiplier) = if let Some(number) = value.strip_suffix("ms") {
        (number, 1.0)
    } else if let Some(number) = value.strip_suffix('s') {
        (number, 1_000.0)
    } else if let Some(number) = value.strip_suffix('m') {
        (number, 60_000.0)
    } else if let Some(number) = value.strip_suffix('h') {
        (number, 3_600_000.0)
    } else {
        (value, 1.0)
    };
    let parsed = number.trim().parse::<f64>().map_err(|err| {
        ::lashlang::ExecutionHostError::new(format!("invalid duration `{value}`: {err}"))
    })?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err(::lashlang::ExecutionHostError::new(format!(
            "invalid non-negative duration `{value}`"
        )));
    }
    Ok((parsed * multiplier).round() as u64)
}

fn process_id_from_lashlang_handle(
    handle: &::lashlang::Value,
) -> Result<String, ::lashlang::ExecutionHostError> {
    let value = crate::lashlang_bridge::lashlang_value_to_json(handle)?;
    let kind = value
        .get("__handle__")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            ::lashlang::ExecutionHostError::new("signal run expects a process handle")
        })?;
    if kind != "process" {
        return Err(::lashlang::ExecutionHostError::new(format!(
            "signal run expects a process handle, got `{kind}`"
        )));
    }
    value
        .get("id")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            ::lashlang::ExecutionHostError::new("signal run process handle is missing `id`")
        })
}

fn process_lashlang_execution_result(
    result: Result<::lashlang::ExecutionOutcome, ::lashlang::RuntimeError>,
) -> crate::ProcessAwaitOutput {
    match result {
        Ok(::lashlang::ExecutionOutcome::Finished(value)) => crate::ProcessAwaitOutput::Success {
            value: crate::lashlang_bridge::lashlang_value_to_json(&value)
                .unwrap_or(serde_json::Value::Null),
            control: None,
        },
        Ok(::lashlang::ExecutionOutcome::Failed(value)) => process_lashlang_failure(
            "process_failed",
            value.to_string(),
            Some(
                crate::lashlang_bridge::lashlang_value_to_json(&value)
                    .unwrap_or(serde_json::Value::Null),
            ),
        ),
        Ok(::lashlang::ExecutionOutcome::Continued) => crate::ProcessAwaitOutput::Success {
            value: serde_json::Value::Null,
            control: None,
        },
        Err(err) => process_lashlang_failure("process_runtime_error", err.to_string(), None),
    }
}

fn process_lashlang_failure(
    code: &str,
    message: String,
    raw: Option<serde_json::Value>,
) -> crate::ProcessAwaitOutput {
    crate::ProcessAwaitOutput::Failure {
        class: crate::ToolFailureClass::Execution,
        code: code.to_string(),
        message,
        raw,
        control: None,
    }
}

fn process_lashlang_cancelled(message: impl Into<String>) -> crate::ProcessAwaitOutput {
    crate::ProcessAwaitOutput::Cancelled {
        message: message.into(),
        raw: None,
        control: None,
    }
}
