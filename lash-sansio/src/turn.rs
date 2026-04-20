use crate::mode::ModePreamble;
use crate::prompt::{PreparedPrompt, PromptBuildInput, build_prompt};
use crate::sansio::{RlmTermination, TurnMachine, TurnMachineConfig};
use crate::{MessageSequence, PromptContribution, PromptTemplate, ToolSurface};

pub struct SansIoTurnInput {
    pub session_id: String,
    pub run_session_id: Option<String>,
    pub model: String,
    pub mode: crate::ExecutionMode,
    pub messages: MessageSequence,
    pub run_offset: usize,
    pub mode_preamble: ModePreamble,
    pub tool_surface: ToolSurface,
    pub prompt_template: PromptTemplate,
    pub prompt_contributions: Vec<PromptContribution>,
    pub max_turns: Option<usize>,
    pub model_variant: Option<String>,
    pub emit_llm_debug_log: bool,
    pub rlm_termination: RlmTermination,
}

pub struct PreparedTurnMachine {
    pub machine: TurnMachine,
    pub prepared_prompt: PreparedPrompt,
    pub tool_surface: ToolSurface,
    pub mode_preamble: ModePreamble,
}

pub fn build_turn(input: SansIoTurnInput) -> PreparedTurnMachine {
    let mut prompt_contributions = input.mode_preamble.prompt_contributions.clone();
    prompt_contributions.extend(input.prompt_contributions);
    let prepared_prompt = build_prompt(PromptBuildInput {
        mode: input.mode,
        template: input.prompt_template,
        execution_prompt: input.mode_preamble.execution_prompt.clone(),
        tool_names: input.mode_preamble.tool_names.clone(),
        omitted_tool_count: input.mode_preamble.omitted_tool_count,
        contributions: prompt_contributions,
    });

    let machine = TurnMachine::new_shared(
        TurnMachineConfig {
            protocol_driver: input.mode_preamble.config.protocol.clone(),
            sync_execution_surface: input.mode_preamble.config.sync_execution_surface,
            model: input.model,
            max_turns: input.max_turns,
            model_variant: input.model_variant,
            run_session_id: input.run_session_id,
            tool_specs: input.mode_preamble.tool_specs.clone(),
            system_prompt: prepared_prompt.system_prompt.clone(),
            session_id: input.session_id,
            emit_llm_debug_log: input.emit_llm_debug_log,
            rlm_termination: input.rlm_termination,
        },
        input.messages,
        input.run_offset,
    );

    PreparedTurnMachine {
        machine,
        prepared_prompt,
        tool_surface: input.tool_surface,
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
        ExecutionMode, PromptContribution, ToolDefinition, ToolExecutionMode, ToolParam,
        default_prompt_template,
    };

    fn tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: format!("Tool {name}"),
            params: vec![ToolParam::typed("path", "str")],
            returns: "str".to_string(),
            examples: Vec::new(),
            enabled: true,
            injected: true,
            input_schema_override: None,
            output_schema_override: None,
            execution_mode: ToolExecutionMode::Parallel,
        }
    }

    /// Minimal no-op driver so the turn-machine test can build a
    /// `ModePreamble` without pulling in a mode crate (which would
    /// create a cyclic dependency on `lash` from `lash-sansio`).
    struct NoopDriver;

    impl ProtocolDriverHandle for NoopDriver {
        fn prepare_iteration(&self, _ctx: DriverContextView<'_>) -> Vec<DriverAction> {
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
        let tool_surface = ToolSurface::from_tools(vec![tool("read_file")]);
        let mode_preamble = ModePreamble {
            config: ModeConfig {
                protocol: Arc::new(NoopDriver),
                sync_execution_surface: false,
            },
            tool_specs: Arc::new(tool_surface.model_tool_specs()),
            tool_names: tool_surface.tool_names(),
            omitted_tool_count: 0,
            execution_prompt: "test prompt".to_string(),
            prompt_contributions: Vec::new(),
        };
        let prepared = build_turn(SansIoTurnInput {
            session_id: "session".to_string(),
            run_session_id: Some("run".to_string()),
            model: "gpt-5".to_string(),
            mode: ExecutionMode::Standard,
            messages: crate::MessageSequence::default(),
            run_offset: 2,
            mode_preamble,
            tool_surface,
            prompt_template: default_prompt_template(),
            prompt_contributions: vec![PromptContribution::guidance("Guide", "Be precise.")],
            max_turns: Some(3),
            model_variant: Some("mini".to_string()),
            emit_llm_debug_log: true,
            rlm_termination: RlmTermination::default(),
        });

        assert_eq!(prepared.machine.iteration(), 2);
        assert!(
            prepared
                .prepared_prompt
                .system_prompt
                .contains("Be precise.")
        );
        assert_eq!(prepared.mode_preamble.tool_specs.len(), 1);
    }
}
