use super::*;
use std::sync::Arc;

#[async_trait::async_trait]
impl crate::runtime::effect::ProcessRunner for RuntimeSessionServices {
    async fn run_process(
        &self,
        registration: crate::ProcessRegistration,
        execution_context: crate::ProcessExecutionContext,
        registry: Arc<dyn crate::ProcessRegistry>,
        scoped_effect_controller: crate::ScopedEffectController<'_>,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::ProcessAwaitOutput {
        let input = Arc::clone(&registration.input);
        // Hybrid process model by design:
        // - ToolCall, SessionTurn, and External are kernel primitives because
        //   core owns their orchestration contracts directly.
        // - Engine rows are deployment runtimes looked up from the registry.
        // This split keeps core process coordination explicit without pulling
        // language-specific runtimes into the kernel.
        match input.as_ref() {
            crate::ProcessInput::ToolCall { call } => {
                self.run_process_tool_call(ProcessToolCallRun {
                    registration,
                    registry: Arc::clone(&registry),
                    call: call.clone(),
                    parent_invocation: execution_context.causal_invocation,
                    scoped_effect_controller,
                    cancellation,
                })
                .await
            }
            crate::ProcessInput::SessionTurn {
                create_request,
                turn_input,
                ..
            } => {
                self.run_process_session_turn(
                    registration,
                    *create_request.clone(),
                    *turn_input.clone(),
                    scoped_effect_controller,
                    cancellation,
                )
                .await
            }
            crate::ProcessInput::Engine { kind, payload } => {
                let engine = match self.current.host.core.process_engines.require(kind) {
                    Ok(engine) => engine,
                    Err(err) => return process_engine_failure("process_engine_missing", err),
                };
                let engine_context = self.process_engine_run_context(
                    registration,
                    execution_context,
                    registry,
                    scoped_effect_controller,
                    cancellation,
                );
                engine.run(engine_context, payload.clone()).await
            }
            crate::ProcessInput::External { metadata } => crate::ProcessAwaitOutput::Success {
                value: serde_json::json!({ "metadata": metadata.clone() }),
                control: None,
            },
        }
    }
}

impl RuntimeSessionServices {
    fn process_engine_run_context<'run>(
        &self,
        registration: crate::ProcessRegistration,
        execution_context: crate::ProcessExecutionContext,
        registry: Arc<dyn crate::ProcessRegistry>,
        scoped_effect_controller: crate::ScopedEffectController<'run>,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::ProcessEngineRunContext<'run> {
        let session_id = self.current.session_id.clone();
        let plugins = Arc::clone(&self.current.plugins);
        let store = self.current.store.clone();
        let session_store_factory = self.current.host.session_store_factory.clone();
        let queued_work_poke = self.current.host.queued_work_poke.clone();
        let process_registry_available = self.current.host.process_registry.is_some();
        let services = self.clone();
        let registration_for_runtime = registration.clone();
        let execution_context_for_runtime = execution_context.clone();
        let registry_for_runtime = Arc::clone(&registry);
        let cancellation_for_runtime = cancellation.clone();
        let builder = Box::new(move |tool_catalog: Arc<crate::ToolCatalog>| {
            let run_context = ProcessRunContext::builder(&services)
                .tool_catalog(tool_catalog)
                .scoped_effect_controller(scoped_effect_controller)
                .causal_invocation(execution_context_for_runtime.causal_invocation.clone())
                .build()?;
            let mut context = crate::RuntimeExecutionContext::new(
                services.current.session_id.clone(),
                run_context.dispatch(),
                Arc::clone(&services.current.host.core.durability.process_env_store),
                Arc::clone(&services.current.host.core.durability.attachment_store),
                Arc::new(crate::ChronologicalProjection::default()),
                None,
                crate::TurnContext::default(),
            )
            .with_execution_env_spec(current_execution_env_spec(&services.current))
            .with_turn_phase_probe(services.current.turn_phase_probe.clone())
            .with_process_registration_context(&registration_for_runtime)
            .with_process_event_context(
                registration_for_runtime.id.clone(),
                Arc::clone(&registry_for_runtime),
                services.current.store.clone(),
                services.current.host.session_store_factory.clone(),
                services.current.host.queued_work_poke.clone(),
            )
            .with_cancellation_token(cancellation_for_runtime.clone());
            if let Some(invocation) = execution_context_for_runtime.causal_invocation.clone() {
                context = context.with_parent_invocation(invocation);
            }
            let guard = crate::ProcessEngineRunGuard::new(move || {
                Box::pin(async move {
                    run_context.shutdown().await;
                })
            });
            Ok(crate::ProcessEngineRuntimeContext::new(context, guard))
        });
        crate::ProcessEngineRunContext::new(
            registration,
            execution_context,
            registry,
            session_id,
            plugins,
            store,
            session_store_factory,
            queued_work_poke,
            process_registry_available,
            cancellation,
            self.current.turn_phase_probe.clone(),
            builder,
        )
    }
}

fn current_execution_env_spec(
    current: &CurrentSessionCapability,
) -> crate::ProcessExecutionEnvSpec {
    let state = current.snapshot.to_runtime_state();
    state.process_execution_env_spec(&current.policy)
}

fn process_engine_failure(code: &str, err: crate::PluginError) -> crate::ProcessAwaitOutput {
    crate::ProcessAwaitOutput::Failure {
        class: crate::ToolFailureClass::Execution,
        code: code.to_string(),
        message: err.to_string(),
        raw: None,
        control: None,
    }
}
