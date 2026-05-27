use std::num::NonZeroUsize;

use lash_sansio::{PreparedPrompt, PromptCache, PromptContributionSet, PromptLayer};

use super::*;
use crate::{PluginError, ToolSurface, TurnDriverPreamble};

struct PreparedExecutionSurface {
    tool_surface: Arc<ToolSurface>,
    turn_driver_preamble: Arc<TurnDriverPreamble>,
    prompt: PromptLayer,
}

impl PreparedExecutionSurface {
    fn build_prompt(
        &self,
        core_prompt: &PromptLayer,
        session_prompt: &PromptLayer,
        turn_prompt: &PromptLayer,
        prompt_cache: Option<Arc<PromptCache>>,
    ) -> PreparedPrompt {
        let mut capability_prompt = PromptLayer::new();
        for contribution in self
            .turn_driver_preamble
            .prompt_contributions
            .iter()
            .cloned()
        {
            capability_prompt.add_contribution(contribution);
        }
        let resolved = crate::resolve_prompt_layers([
            &capability_prompt,
            &self.prompt,
            core_prompt,
            session_prompt,
            turn_prompt,
        ]);
        let prompt_contributions = self
            .tool_surface
            .filter_prompt_contributions(resolved.contributions);
        let contributions = PromptContributionSet::new(prompt_contributions);
        lash_sansio::build_prompt_cached(
            crate::PromptBuildInput {
                template_fingerprint: crate::prompt_template_fingerprint(&resolved.template),
                template: resolved.template,
                execution_prompt_fingerprint: crate::prompt_text_fingerprint(
                    &self.turn_driver_preamble.execution_prompt,
                ),
                execution_prompt: Arc::clone(&self.turn_driver_preamble.execution_prompt),
                tool_names_fingerprint: self.turn_driver_preamble.tool_names_fingerprint,
                tool_names: Arc::clone(&self.turn_driver_preamble.tool_names),
                omitted_tool_count: self.turn_driver_preamble.omitted_tool_count,
                contributions,
            },
            prompt_cache.as_deref(),
        )
    }
}

fn generation_options_from_provider(provider: &crate::ProviderHandle) -> crate::GenerationOptions {
    crate::GenerationOptions {
        output_token_cap: provider
            .options()
            .max_output_tokens
            .and_then(|value| usize::try_from(value).ok())
            .and_then(NonZeroUsize::new),
    }
}

impl RuntimeTurnDriver<'_> {
    pub(in crate::runtime) fn restore_turn_machine(
        &mut self,
        checkpoint_record: crate::RuntimeTurnCheckpoint,
    ) -> Result<TurnMachine, RuntimeError> {
        let checkpoint_hash = crate::runtime_turn_checkpoint_hash(&checkpoint_record.checkpoint)
            .map_err(|err| {
                RuntimeError::new(RuntimeErrorCode::RuntimeTurnCheckpointHash, err.to_string())
            })?;
        if checkpoint_hash != checkpoint_record.checkpoint_hash {
            return Err(RuntimeError::new(
                RuntimeErrorCode::RuntimeTurnCheckpointHashMismatch,
                format!(
                    "persisted checkpoint hash `{}` does not match decoded checkpoint hash `{}`",
                    checkpoint_record.checkpoint_hash, checkpoint_hash
                ),
            ));
        }
        let turn_driver_preamble = self
            .session
            .turn_driver_preamble(&self.session_id)
            .map_err(|err| {
                RuntimeError::new(
                    RuntimeErrorCode::RuntimeTurnRestoreTurnDriverPreamble,
                    err.to_string(),
                )
            })?;
        self.machine_config_snapshot = Some(checkpoint_record.machine_config.clone());
        self.protocol_turn_options = checkpoint_record.protocol_turn_options.clone();
        self.turn_context
            .set_prompt_layer(checkpoint_record.turn_prompt_layer.clone());
        let config = crate::TurnMachineConfig {
            protocol_driver: turn_driver_preamble.config.protocol.clone(),
            projector: turn_driver_preamble.config.projector.clone(),
            sync_execution_surface: checkpoint_record.machine_config.sync_execution_surface,
            model: checkpoint_record.machine_config.model.id.clone(),
            max_turns: checkpoint_record.machine_config.max_turns,
            model_variant: checkpoint_record.machine_config.model.variant.clone(),
            generation: checkpoint_record.machine_config.generation.clone(),
            run_session_id: checkpoint_record.machine_config.run_session_id.clone(),
            autonomous: checkpoint_record.machine_config.autonomous,
            tool_specs: Arc::new(checkpoint_record.machine_config.tool_specs.clone()),
            system_prompt: Arc::<str>::from(checkpoint_record.machine_config.system_prompt.clone()),
            session_id: checkpoint_record.machine_config.session_id.clone(),
            emit_llm_trace: false,
            termination: checkpoint_record.machine_config.termination.clone(),
            turn_limit_final_message: turn_driver_preamble.config.turn_limit_final_message.clone(),
        };
        Ok(TurnMachine::restore_from_checkpoint(
            config,
            checkpoint_record.checkpoint,
        ))
    }

    pub(super) async fn prepare_turn_machine(
        &mut self,
        messages: crate::MessageSequence,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        run_offset: usize,
    ) -> Result<
        (TurnMachine, crate::RuntimeTurnMachineConfigSnapshot),
        (crate::MessageSequence, usize),
    > {
        macro_rules! emit {
            ($event:expr) => {
                send_session_event(event_tx, $event).await
            };
        }

        let mut session_policy = self.runtime_session_policy();
        let model = match self.prepare_provider(&mut session_policy).await {
            Ok(model) => model,
            Err(event) => {
                emit!(event);
                emit!(SessionEvent::Done);
                return Err((messages.clone(), run_offset));
            }
        };
        let execution_surface = match self
            .prepare_execution_surface(&session_policy, self.turn_index, messages.clone())
            .await
        {
            Ok(surface) => surface,
            Err(err) => {
                emit!(make_error_event(
                    "plugin_prompt",
                    None,
                    err.to_string(),
                    Some(err.to_string()),
                ));
                emit!(SessionEvent::Done);
                return Err((messages, run_offset));
            }
        };
        self.mark_phase_begin(RuntimeTurnPhase::PromptBuild);
        let prepared_prompt = execution_surface.build_prompt(
            &self.host.core.prompt,
            &session_policy.prompt,
            self.turn_context.prompt_layer(),
            Some(self.session.prompt_cache()),
        );
        let machine_config_snapshot = crate::RuntimeTurnMachineConfigSnapshot {
            session_id: self.session_id.clone(),
            run_session_id: session_policy.session_id.clone(),
            autonomous: session_policy.autonomous,
            model: session_policy.model.clone(),
            generation: generation_options_from_provider(&session_policy.provider),
            max_turns: session_policy.max_turns,
            sync_execution_surface: execution_surface
                .turn_driver_preamble
                .config
                .sync_execution_surface,
            tool_specs: execution_surface
                .turn_driver_preamble
                .tool_specs
                .as_ref()
                .clone(),
            system_prompt: prepared_prompt.system_prompt.to_string(),
            termination: self.protocol_turn_options.clone(),
        };
        let prepared = crate::build_turn(crate::SansIoTurnInput {
            session_id: self.session_id.clone(),
            run_session_id: session_policy.session_id.clone(),
            autonomous: session_policy.autonomous,
            model,
            messages,
            events: self.turn_pipeline.active_events(),
            protocol_run_offset: run_offset,
            turn_driver_preamble: execution_surface.turn_driver_preamble,
            prepared_prompt,
            max_turns: session_policy.max_turns,
            model_variant: session_policy.model.variant.clone(),
            generation: generation_options_from_provider(&session_policy.provider),
            emit_llm_trace: false,
            termination: self.protocol_turn_options.clone(),
        });
        if self.host.core.trace_sink.is_some() {
            let prompt_hash =
                lash_trace::sha256_hex(prepared.prepared_prompt.system_prompt.as_bytes());
            let prompt_chars = prepared.prepared_prompt.system_prompt.chars().count();
            crate::trace::emit_trace(
                &self.host.core.trace_sink,
                &self.host.core.trace_context,
                self.trace_context(run_offset),
                lash_trace::TraceEvent::PromptBuilt {
                    prompt_hash: prompt_hash.clone(),
                    prompt_chars,
                    components: vec![lash_trace::TracePromptComponent {
                        id: "system_prompt".to_string(),
                        kind: "rendered_prompt".to_string(),
                        hash: prompt_hash,
                        chars: Some(prompt_chars),
                    }],
                },
            );
        }
        self.policy = session_policy;
        self.mark_phase_end(RuntimeTurnPhase::PromptBuild);
        Ok((prepared.machine, machine_config_snapshot))
    }

    pub(in crate::runtime) async fn refresh_execution_surface(
        &mut self,
        machine: &crate::TurnMachine,
        update_machine_config: bool,
    ) -> Result<Option<crate::sansio::ExecutionSurfaceSync>, crate::SessionError> {
        if !update_machine_config {
            return Ok(None);
        }

        let policy = self.policy.clone();
        let execution_surface = self
            .prepare_execution_surface(&policy, self.turn_index, machine.message_sequence())
            .await
            .map_err(|err| crate::SessionError::Protocol(err.to_string()))?;
        let prepared_prompt = execution_surface.build_prompt(
            &self.host.core.prompt,
            &policy.prompt,
            self.turn_context.prompt_layer(),
            Some(self.session.prompt_cache()),
        );

        Ok(Some(crate::sansio::ExecutionSurfaceSync {
            system_prompt: prepared_prompt.system_prompt,
            tool_specs: execution_surface.turn_driver_preamble.tool_specs.clone(),
        }))
    }

    async fn prepare_execution_surface(
        &mut self,
        session_policy: &SessionPolicy,
        turn_index: usize,
        messages: crate::MessageSequence,
    ) -> Result<PreparedExecutionSurface, PluginError> {
        let tool_surface = self.session.tool_surface(&self.session_id)?;
        let turn_driver_preamble = self.session.turn_driver_preamble(&self.session_id)?;
        let plugin_prompt_contributions = self
            .session
            .plugins()
            .collect_prompt_contributions(PromptHookContext {
                session_id: self.session_id.clone(),
                host: self.session_manager.clone(),
                state: self.turn_pipeline.read_view(
                    session_policy.clone(),
                    turn_index,
                    self.protocol_turn_options.clone(),
                    messages,
                ),
                protocol_turn_options: self.protocol_turn_options.clone(),
                turn_context: self.turn_context.clone(),
            })
            .await?;
        let mut prompt = PromptLayer::new();
        for contribution in self.session.context_prompt_contributions().iter().cloned() {
            prompt.add_contribution(contribution);
        }
        for contribution in plugin_prompt_contributions {
            prompt.add_contribution(contribution);
        }
        if let Some(extension) = &self.protocol_extension {
            for contribution in extension.prompt_contributions() {
                prompt.add_contribution(contribution);
            }
        }
        Ok(PreparedExecutionSurface {
            tool_surface,
            turn_driver_preamble,
            prompt,
        })
    }

    pub(super) async fn before_llm_call(
        &mut self,
        machine: &TurnMachine,
        request: &LlmRequest,
    ) -> Result<Option<crate::ProtocolLlmCallAction>, PluginError> {
        let latest_prompt_usage = self.turn_pipeline.state_mut().last_prompt_usage.clone();
        let effect_controller =
            crate::runtime::RuntimeEffectControllerHandle::borrowed(self.effect_scope.controller());
        let direct_completions = self.session_manager.direct_completion_client(
            effect_controller.clone_scoped(),
            Some(self.turn_id.clone()),
            self.turn_lease.clone(),
        );
        let process_effect_metadata = self.turn_effect_metadata(
            machine,
            crate::sansio::EffectId(u64::MAX),
            RuntimeEffectKind::Process,
        )?;
        self.session
            .plugins()
            .protocol_session()
            .before_llm_call(
                crate::ProtocolBeforeLlmCallContext {
                    session_id: self.session_id.clone(),
                    host: self.session_manager.clone()
                        as Arc<dyn crate::plugin::RuntimeSessionHost>,
                    processes: self.session_manager.clone() as Arc<dyn crate::ProcessService>,
                    state: self.checkpoint_state_view(
                        machine.message_sequence(),
                        machine.protocol_iteration(),
                    ),
                    latest_prompt_usage,
                    direct_completions,
                    process_effect_metadata,
                    effect_controller,
                },
                request,
            )
            .await
    }

    fn runtime_session_policy(&self) -> SessionPolicy {
        self.policy.clone()
    }

    pub(super) fn checkpoint_state_view(
        &self,
        messages: crate::MessageSequence,
        _protocol_iteration: usize,
    ) -> crate::SessionReadView {
        self.turn_pipeline.read_view(
            self.policy.clone(),
            self.turn_index,
            self.protocol_turn_options.clone(),
            messages,
        )
    }

    pub(super) async fn prepare_provider(
        &mut self,
        policy: &mut SessionPolicy,
    ) -> Result<String, SessionEvent> {
        let model = policy.model.id.clone();
        if let Some(variant) = policy.model.variant.as_deref()
            && let Err(message) = policy.provider.validate_variant(&model, variant)
        {
            return Err(make_error_event(
                "llm_provider",
                Some("invalid_model_variant"),
                message.clone(),
                Some(message),
            ));
        }
        Ok(model)
    }
}
