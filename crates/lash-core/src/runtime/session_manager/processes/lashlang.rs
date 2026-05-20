use super::*;
use std::collections::HashMap;
use std::sync::Arc;

impl RuntimeSessionManager {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime::session_manager::processes) async fn run_lashlang_process(
        &self,
        registration: crate::ProcessRegistration,
        registry: Arc<dyn crate::ProcessRegistry>,
        program: serde_json::Value,
        input: serde_json::Map<String, serde_json::Value>,
        tool_bindings: Vec<crate::LashlangProcessToolBinding>,
        timeout_ms: Option<u64>,
        execution_context: crate::ProcessExecutionContext,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::ProcessAwaitOutput {
        let program = match serde_json::from_value::<::lashlang::Program>(program) {
            Ok(program) => program,
            Err(err) => {
                return process_lashlang_failure(
                    "process_block_decode_failed",
                    format!("failed to decode lashlang process block: {err}"),
                    None,
                );
            }
        };
        let mut globals = ::lashlang::Record::with_capacity(input.len());
        for (name, value) in input {
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
                    "process_block_tool_surface_failed",
                    err.to_string(),
                    None,
                );
            }
        };
        let mut ctx = crate::ModeExecutionContext::new(
            self.current.session_id.clone(),
            self.current.policy.execution_mode.clone(),
            Arc::new(dispatch),
            Arc::clone(&self.current.host.core.attachment_store),
            Arc::new(crate::ChronologicalProjection::default()),
            None,
            crate::TurnContext::default(),
        )
        .with_cancellation_token(cancellation.clone());
        if let Some(metadata) = execution_context.tool_effect_metadata.clone() {
            ctx = ctx.with_effect_metadata(metadata);
        }

        let host = LashlangBlockProcessHost {
            manager: self.clone(),
            ctx,
            registry: Arc::clone(&registry),
            process_id: registration.id.clone(),
            tool_bindings: tool_bindings
                .into_iter()
                .map(|binding| (binding.name, binding.tool_id))
                .collect(),
            wake_session_id: execution_context.wake_session_id,
        };
        let env = ::lashlang::ExecutionEnvironment::new(&host).process();

        let output = if let Some(timeout_ms) = timeout_ms {
            let timeout = tokio::time::Duration::from_millis(timeout_ms);
            tokio::select! {
                _ = cancellation.cancelled() => process_lashlang_cancelled("lashlang process block was cancelled"),
                result = tokio::time::timeout(timeout, ::lashlang::execute(&program, &mut state, &env)) => {
                    match result {
                        Ok(result) => process_lashlang_execution_result(result),
                        Err(_) => process_lashlang_failure(
                            "process_block_timeout",
                            format!("lashlang process block timed out after {timeout_ms}ms"),
                            None,
                        ),
                    }
                }
            }
        } else {
            tokio::select! {
                _ = cancellation.cancelled() => process_lashlang_cancelled("lashlang process block was cancelled"),
                result = ::lashlang::execute(&program, &mut state, &env) => {
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

struct LashlangBlockProcessHost<'run> {
    manager: RuntimeSessionManager,
    ctx: crate::ModeExecutionContext<'run>,
    registry: Arc<dyn crate::ProcessRegistry>,
    process_id: String,
    tool_bindings: HashMap<String, crate::ToolId>,
    wake_session_id: Option<String>,
}

impl LashlangBlockProcessHost<'_> {
    fn captured_manifest(
        &self,
        name: &str,
    ) -> Result<crate::ToolManifest, ::lashlang::ExecutionHostError> {
        let tool_id = self.tool_bindings.get(name).ok_or_else(|| {
            ::lashlang::ExecutionHostError::new(format!(
                "tool `{name}` was not captured for this lashlang process"
            ))
        })?;
        self.ctx
            .callable_tool_manifest_by_id(tool_id)
            .ok_or_else(|| {
                ::lashlang::ExecutionHostError::new(format!(
                    "captured tool `{name}` with id `{}` is unavailable in this session",
                    tool_id.as_str()
                ))
            })
    }

    fn tool_payload(
        &self,
        args: &::lashlang::Record,
    ) -> Result<serde_json::Value, ::lashlang::ExecutionHostError> {
        let mut payload = crate::lashlang_bridge::lashlang_value_to_json(
            &::lashlang::Value::Record(std::sync::Arc::new(args.clone())),
        )?;
        if let Some(obj) = payload.as_object_mut() {
            obj.entry("__session_id__".to_string())
                .or_insert_with(|| serde_json::Value::String(self.ctx.session_id().to_string()));
        }
        Ok(payload)
    }
}

impl LashlangBlockProcessHost<'_> {
    async fn call(
        &self,
        name: String,
        args: ::lashlang::Record,
    ) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        let manifest = self.captured_manifest(&name)?;
        let reply = self
            .ctx
            .call_tool(
                uuid::Uuid::new_v4().to_string(),
                manifest.name.clone(),
                self.tool_payload(&args)?,
                0,
            )
            .await;
        mode_reply_to_lashlang_value(reply)
    }

    async fn start_call(
        &self,
        name: String,
        args: ::lashlang::Record,
    ) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        let manifest = self.captured_manifest(&name)?;
        let reply = self
            .ctx
            .start_tool_call(
                uuid::Uuid::new_v4().to_string(),
                manifest.name.clone(),
                self.tool_payload(&args)?,
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
        start: ::lashlang::ProcessBlockStart,
    ) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        let (registration, label) = self
            .ctx
            .prepare_lashlang_process_start(start)
            .map_err(::lashlang::ExecutionHostError::new)?;
        let reply = self.ctx.start_lashlang_process(registration, label).await;
        mode_reply_to_lashlang_value(reply)
    }

    async fn process_event(
        &self,
        event: ::lashlang::ProcessBlockEvent,
    ) -> Result<(), ::lashlang::ExecutionHostError> {
        let event_type = match event.kind {
            ::lashlang::ProcessBlockEventKind::Yield => "process.yield",
            ::lashlang::ProcessBlockEventKind::Wake => "process.wake",
        };
        let event = self
            .registry
            .append_event(
                &self.process_id,
                event_type.to_string(),
                crate::lashlang_bridge::process_event_payload(&event.value)?,
            )
            .await
            .map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?;
        if let Some(wake) = event.semantics.wake.as_ref()
            && self
                .manager
                .managed
                .inject_turn_input(
                    self.wake_session_id
                        .as_deref()
                        .unwrap_or_else(|| self.ctx.session_id()),
                    crate::InjectedTurnInput {
                        id: Some(format!(
                            "process:{}:wake:{}",
                            self.process_id, event.sequence
                        )),
                        message: crate::PluginMessage::text(
                            crate::MessageRole::System,
                            wake.input.clone(),
                        ),
                    },
                )
                .await
                .is_ok()
        {
            self.registry
                .ack_wake(&self.process_id, event.sequence)
                .await
                .map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?;
        }
        Ok(())
    }
}

impl ::lashlang::ExecutionHost for LashlangBlockProcessHost<'_> {
    async fn perform(
        &self,
        op: ::lashlang::AbilityOp,
    ) -> Result<::lashlang::AbilityResult, ::lashlang::ExecutionHostError> {
        match op {
            ::lashlang::AbilityOp::CallTool { name, args } => self
                .call(name, args)
                .await
                .map(::lashlang::AbilityResult::Value),
            ::lashlang::AbilityOp::StartToolCall { name, args } => self
                .start_call(name, args)
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
            ::lashlang::AbilityOp::Print(_) => Err(::lashlang::ExecutionHostError::new(
                "`print` is not available inside lashlang process blocks",
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
            "process_block_failed",
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
        Err(err) => process_lashlang_failure("process_block_runtime_error", err.to_string(), None),
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
