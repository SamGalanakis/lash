use std::sync::Arc;

use crate::mode::ModePreamble;
use crate::prompt::{PreparedPrompt, PromptBuildInput, PromptCache, build_prompt_cached};
use crate::sansio::{ModeProtocol, TurnMachine, TurnMachineConfig, UnitModeProtocol};
use crate::session_model::RetryPolicy;
use crate::{MessageSequence, PromptContribution, PromptTemplate, ToolSurface};

pub struct SansIoTurnInput<M: ModeProtocol = UnitModeProtocol> {
    pub session_id: String,
    pub run_session_id: Option<String>,
    pub autonomous: bool,
    pub model: String,
    pub mode: crate::ExecutionMode,
    pub messages: MessageSequence,
    pub events: Arc<Vec<crate::SessionEventRecord<M::Event>>>,
    pub mode_run_offset: usize,
    pub tool_surface: Arc<ToolSurface>,
    pub mode_preamble: Arc<ModePreamble<M>>,
    pub prompt_template: PromptTemplate,
    pub prompt_contributions: Vec<PromptContribution>,
    pub max_turns: Option<usize>,
    pub model_variant: Option<String>,
    pub emit_llm_trace: bool,
    pub termination: M::Termination,
    pub retry_policy: RetryPolicy,
    /// Optional cache that lets repeat turns skip re-rendering the
    /// system prompt when inputs are unchanged.
    pub prompt_cache: Option<Arc<PromptCache>>,
}

pub struct PreparedTurnMachine<M: ModeProtocol = UnitModeProtocol> {
    pub machine: TurnMachine<M>,
    pub prepared_prompt: PreparedPrompt,
    pub mode_preamble: Arc<ModePreamble<M>>,
}

pub fn build_turn<M: ModeProtocol>(input: SansIoTurnInput<M>) -> PreparedTurnMachine<M> {
    let mut prompt_contributions = input.mode_preamble.prompt_contributions.clone();
    prompt_contributions.extend(input.prompt_contributions);
    let prompt_contributions = input
        .tool_surface
        .filter_prompt_contributions(prompt_contributions);
    let prepared_prompt = build_prompt_cached(
        PromptBuildInput {
            mode: input.mode,
            template: input.prompt_template,
            execution_prompt: input.mode_preamble.execution_prompt.clone(),
            tool_names: input.mode_preamble.tool_names.clone(),
            omitted_tool_count: input.mode_preamble.omitted_tool_count,
            contributions: prompt_contributions,
        },
        input.prompt_cache.as_deref(),
    );

    let machine = TurnMachine::new_shared(
        TurnMachineConfig {
            protocol_driver: input.mode_preamble.config.protocol.clone(),
            projector: input.mode_preamble.config.projector.clone(),
            sync_execution_surface: input.mode_preamble.config.sync_execution_surface,
            model: input.model,
            max_turns: input.max_turns,
            model_variant: input.model_variant,
            run_session_id: input.run_session_id,
            autonomous: input.autonomous,
            tool_specs: input.mode_preamble.tool_specs.clone(),
            system_prompt: Arc::clone(&prepared_prompt.system_prompt),
            session_id: input.session_id,
            emit_llm_trace: input.emit_llm_trace,
            termination: input.termination,
            retry_policy: input.retry_policy,
        },
        input.messages,
        input.events,
        input.mode_run_offset,
    );

    PreparedTurnMachine {
        machine,
        prepared_prompt,
        mode_preamble: input.mode_preamble,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::mode::{ModeConfig, ModePreamble};
    use crate::sansio::{
        CompletedToolCall, DriverAction, DriverContextView, ProtocolDriverHandle, WaitingExecState,
        WaitingLlmState,
    };
    use crate::{
        ExecutionMode, PromptContribution, ToolDefinition, ToolExecutionMode,
        default_prompt_template,
    };

    fn tool(name: &str) -> ToolDefinition {
        let mut definition = ToolDefinition::new(
            name,
            format!("Tool {name}"),
            serde_json::json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
            serde_json::json!({ "type": "string" }),
        );
        definition.execution_mode = ToolExecutionMode::Parallel;
        definition
    }

    /// Minimal no-op driver so the turn-machine test can build a
    /// `ModePreamble` without pulling in a mode crate (which would
    /// create a cyclic dependency on `lash` from `lash-sansio`).
    struct NoopDriver;

    impl ProtocolDriverHandle for NoopDriver {
        fn prepare_mode_iteration(&self, _ctx: DriverContextView<'_>) -> Vec<DriverAction> {
            Vec::new()
        }

        fn handle_llm_success(
            &self,
            _ctx: DriverContextView<'_>,
            _waiting: WaitingLlmState,
            _llm_response: crate::llm::types::LlmResponse,
            _text_streamed: bool,
        ) -> Vec<DriverAction> {
            Vec::new()
        }

        fn handle_tool_results(
            &self,
            _ctx: DriverContextView<'_>,
            _completed: Vec<CompletedToolCall>,
        ) -> Vec<DriverAction> {
            Vec::new()
        }

        fn handle_exec_result(
            &self,
            _ctx: DriverContextView<'_>,
            _waiting: WaitingExecState,
            _result: Result<crate::ExecResponse, String>,
        ) -> Vec<DriverAction> {
            Vec::new()
        }
    }

    #[test]
    fn build_turn_creates_machine_with_rendered_system_prompt() {
        let tool_surface = Arc::new(crate::ToolSurface::from_tools(
            vec![tool("read_file")],
            ExecutionMode::standard(),
        ));
        let mode_preamble = Arc::new(ModePreamble {
            config: ModeConfig::chat(Arc::new(NoopDriver), false),
            tool_specs: Arc::new(tool_surface.model_tool_specs()),
            tool_names: tool_surface.tool_names(),
            omitted_tool_count: 0,
            execution_prompt: "test prompt".to_string(),
            prompt_contributions: Vec::new(),
        });
        let prepared = build_turn(SansIoTurnInput {
            session_id: "session".to_string(),
            run_session_id: Some("run".to_string()),
            autonomous: false,
            model: "gpt-5".to_string(),
            mode: ExecutionMode::standard(),
            messages: crate::MessageSequence::default(),
            events: Arc::new(Vec::new()),
            mode_run_offset: 2,
            tool_surface: Arc::clone(&tool_surface),
            mode_preamble,
            prompt_template: default_prompt_template(),
            prompt_contributions: vec![PromptContribution::guidance("Guide", "Be precise.")],
            max_turns: Some(3),
            model_variant: Some("mini".to_string()),
            emit_llm_trace: true,
            termination: (),
            retry_policy: RetryPolicy::default(),
            prompt_cache: None,
        });

        assert_eq!(prepared.machine.mode_iteration(), 2);
        assert!(
            prepared
                .prepared_prompt
                .system_prompt
                .contains("Be precise.")
        );
        assert_eq!(prepared.mode_preamble.tool_specs.len(), 1);
    }
}
