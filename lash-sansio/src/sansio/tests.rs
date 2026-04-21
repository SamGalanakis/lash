use std::sync::Arc;

use super::*;
use crate::llm::types::{LlmOutputPart, LlmRequest, LlmResponse};
use crate::session_model::message::{PartAttachment, data_url_for_bytes};
use crate::session_model::{Message, MessageRole, MessageSequence, Part, PartKind, PruneState};

fn test_config(protocol_driver: Arc<dyn ProtocolDriverHandle>) -> TurnMachineConfig {
    TurnMachineConfig {
        protocol_driver,
        sync_execution_surface: false,
        model: "test-model".to_string(),
        max_turns: None,
        model_variant: None,
        run_session_id: None,
        tool_specs: Vec::new().into(),
        system_prompt: String::new(),
        session_id: "test".to_string(),
        emit_llm_debug_log: false,
        rlm_termination: RlmTermination::default(),
        retry_policy: RetryPolicy::default(),
    }
}

fn user_message(content: &str) -> Message {
    Message {
        id: "m0".to_string(),
        role: MessageRole::User,
        parts: vec![Part {
            id: "m0.p0".to_string(),
            kind: PartKind::Text,
            content: content.to_string(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_item_id: None,
            tool_signature: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
        }],
        user_input: None,
        origin: None,
    }
}

fn text_message(role: MessageRole, content: impl Into<String>) -> Message {
    let id = fresh_message_id();
    Message {
        id: id.clone(),
        role,
        parts: vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Text,
            content: content.into(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_item_id: None,
            tool_signature: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
        }],
        user_input: None,
        origin: None,
    }
}

fn drain_effects(machine: &mut TurnMachine) -> Vec<Effect> {
    let mut effects = Vec::new();
    while let Some(effect) = machine.poll_effect() {
        effects.push(effect);
    }
    effects
}

fn find_llm_call(effects: &[Effect]) -> Option<(&EffectId, &LlmRequest)> {
    effects.iter().find_map(|effect| match effect {
        Effect::LlmCall { id, request } => Some((id, request)),
        _ => None,
    })
}

fn find_exec_call(effects: &[Effect]) -> Option<(&EffectId, &str)> {
    effects.iter().find_map(|effect| match effect {
        Effect::ExecCode { id, code } => Some((id, code.as_str())),
        _ => None,
    })
}

fn find_checkpoint(effects: &[Effect]) -> Option<(EffectId, CheckpointKind)> {
    effects.iter().find_map(|effect| match effect {
        Effect::Checkpoint { id, checkpoint } => Some((*id, *checkpoint)),
        _ => None,
    })
}

fn find_done(effects: &[Effect]) -> Option<(&MessageSequence, usize)> {
    effects.iter().find_map(|effect| match effect {
        Effect::Done {
            messages,
            iteration,
        } => Some((messages, *iteration)),
        _ => None,
    })
}

fn find_sleep(effects: &[Effect]) -> Option<EffectId> {
    effects.iter().find_map(|effect| match effect {
        Effect::Sleep { id, .. } => Some(*id),
        _ => None,
    })
}

struct ProseDriver;

impl ProtocolDriverHandle for ProseDriver {
    fn prepare_iteration(&self, ctx: DriverContextView<'_>) -> Vec<DriverAction> {
        vec![DriverAction::StartLlm {
            request: ctx.build_llm_request(false),
            driver_state: None,
        }]
    }

    fn handle_llm_success(
        &self,
        _ctx: DriverContextView<'_>,
        _waiting: WaitingLlmState,
        _llm_response: LlmResponse,
        _text_streamed: bool,
    ) -> Vec<DriverAction> {
        vec![
            DriverAction::AppendMessages(vec![text_message(MessageRole::Assistant, "done")]),
            DriverAction::StartCheckpoint {
                checkpoint: CheckpointKind::BeforeCompletion,
                on_empty: CheckpointResumeAction::Finish,
            },
        ]
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

struct RetryStateDriver;

impl ProtocolDriverHandle for RetryStateDriver {
    fn prepare_iteration(&self, ctx: DriverContextView<'_>) -> Vec<DriverAction> {
        vec![DriverAction::StartLlm {
            request: ctx.build_llm_request(false),
            driver_state: Some(driver_state(7usize)),
        }]
    }

    fn handle_llm_success(
        &self,
        _ctx: DriverContextView<'_>,
        mut waiting: WaitingLlmState,
        _llm_response: LlmResponse,
        _text_streamed: bool,
    ) -> Vec<DriverAction> {
        let marker = waiting.take_driver_state::<usize>().expect("driver state");
        vec![
            DriverAction::AppendMessages(vec![text_message(
                MessageRole::Assistant,
                format!("state={marker}"),
            )]),
            DriverAction::StartCheckpoint {
                checkpoint: CheckpointKind::BeforeCompletion,
                on_empty: CheckpointResumeAction::Finish,
            },
        ]
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

struct ExecDriver;

impl ProtocolDriverHandle for ExecDriver {
    fn prepare_iteration(&self, _ctx: DriverContextView<'_>) -> Vec<DriverAction> {
        vec![DriverAction::StartExec {
            code: "print 1".to_string(),
            driver_state: driver_state("exec-state".to_string()),
        }]
    }

    fn handle_llm_success(
        &self,
        _ctx: DriverContextView<'_>,
        _waiting: WaitingLlmState,
        _llm_response: LlmResponse,
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
        waiting: WaitingExecState,
        _result: Result<crate::ExecResponse, String>,
    ) -> Vec<DriverAction> {
        let state = waiting
            .into_driver_state::<String>()
            .expect("exec driver state");
        vec![
            DriverAction::AppendMessages(vec![text_message(MessageRole::User, state)]),
            DriverAction::StartCheckpoint {
                checkpoint: CheckpointKind::BeforeCompletion,
                on_empty: CheckpointResumeAction::Finish,
            },
        ]
    }
}

#[test]
fn llm_request_includes_image_prompt_parts_for_attached_images() {
    let config = test_config(Arc::new(ProseDriver));
    let msgs = vec![Message {
        id: "m0".to_string(),
        role: MessageRole::User,
        parts: vec![
            Part {
                id: "m0.p0".to_string(),
                kind: PartKind::Image,
                content: String::new(),
                attachment: Some(PartAttachment {
                    mime: "image/png".to_string(),
                    url: data_url_for_bytes("image/png", &[1, 2, 3]),
                    filename: None,
                }),
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
            },
            Part {
                id: "m0.p1".to_string(),
                kind: PartKind::Text,
                content: "explain this".to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
            },
        ],
        user_input: None,
        origin: None,
    }];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let (_, request) = find_llm_call(&effects).expect("llm call");

    use crate::llm::types::LlmContentBlock;
    assert_eq!(request.attachments.len(), 1);
    assert!(request.messages.iter().any(|msg| {
        msg.blocks
            .iter()
            .any(|block| matches!(block, LlmContentBlock::Image { attachment_idx: 0 }))
    }));
    assert!(request.messages.iter().any(|msg| {
        msg.blocks.iter().any(|block| match block {
            LlmContentBlock::Text(text) => text.contains("explain this"),
            _ => false,
        })
    }));
}

#[test]
fn driver_can_finish_via_checkpoint() {
    let config = test_config(Arc::new(ProseDriver));
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call").0;
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "Hello".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Hello".to_string(),
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: Vec::new(),
        transient_messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_done(&effects).is_some());
    assert!(machine.is_done());
}

#[test]
fn retryable_error_preserves_driver_state_across_retry() {
    let config = test_config(Arc::new(RetryStateDriver));
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call").0;
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Err(LlmCallError {
            message: "rate limited".to_string(),
            retryable: true,
            raw: None,
            code: None,
            request_body: None,
        }),
    });

    let effects = drain_effects(&mut machine);
    let sleep_id = find_sleep(&effects).expect("retry sleep");
    machine.handle_response(Response::Timeout { id: sleep_id });

    let effects = drain_effects(&mut machine);
    let retried_id = *find_llm_call(&effects).expect("retried llm").0;
    machine.handle_response(Response::LlmComplete {
        id: retried_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "ok".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "ok".to_string(),
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, _) = find_checkpoint(&effects).expect("checkpoint");
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: Vec::new(),
        transient_messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    let (messages, _) = find_done(&effects).expect("done");
    assert!(
        messages
            .iter()
            .any(|message| { message.parts.iter().any(|part| part.content == "state=7") })
    );
}

#[test]
fn checkpoint_messages_resume_prepare_iteration() {
    let config = test_config(Arc::new(ProseDriver));
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call").0;
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "Hello".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Hello".to_string(),
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: vec![PluginMessage::text(MessageRole::User, "one more thing")],
        transient_messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_llm_call(&effects).is_some());
    assert!(machine.messages().iter().any(|message| {
        message.role == MessageRole::User
            && message
                .parts
                .iter()
                .any(|part| part.content == "one more thing")
    }));
}

#[test]
fn exec_driver_state_round_trip() {
    let config = test_config(Arc::new(ExecDriver));
    let msgs = vec![user_message("run code")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let (exec_id, code) = find_exec_call(&effects).expect("exec call");
    assert_eq!(code, "print 1");
    machine.handle_response(Response::ExecResult {
        id: *exec_id,
        result: Ok(crate::ExecResponse {
            output: String::new(),
            observations: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            error: None,
            duration_ms: 0,
            terminal_finish: None,
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: Vec::new(),
        transient_messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    let (messages, _) = find_done(&effects).expect("done");
    assert!(messages.iter().any(|message| {
        message
            .parts
            .iter()
            .any(|part| part.content == "exec-state")
    }));
}
