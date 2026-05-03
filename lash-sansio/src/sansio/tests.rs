use std::sync::Arc;

use super::*;
use crate::llm::types::{LlmOutputPart, LlmRequest, LlmResponse};
use crate::session_model::message::PartAttachment;
use crate::session_model::{
    ConversationRecord, Message, MessageRole, MessageSequence, Part, PartKind, PruneState,
    SessionEventRecord,
};

fn test_config(protocol_driver: Arc<dyn ProtocolDriverHandle>) -> TurnMachineConfig {
    TurnMachineConfig {
        protocol_driver,
        projector: Arc::new(ChatContextProjector),
        sync_execution_surface: false,
        model: "test-model".to_string(),
        max_turns: None,
        model_variant: None,
        run_session_id: None,
        autonomous: false,
        tool_specs: Vec::new().into(),
        system_prompt: Arc::from(""),
        session_id: "test".to_string(),
        emit_llm_trace: false,
        termination: (),
        retry_policy: RetryPolicy::default(),
    }
}

fn test_attachment_ref(byte_len: u64) -> crate::AttachmentRef {
    crate::AttachmentRef {
        id: crate::AttachmentId::new("att-test"),
        media_type: crate::MediaType::Image(crate::ImageMediaType::Png),
        byte_len,
        width: None,
        height: None,
        label: None,
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
            response_meta: None,
        }]
        .into(),
        user_input: None,
        origin: None,
    }
}

fn conversation_event(message: Message) -> SessionEventRecord {
    SessionEventRecord::Conversation(ConversationRecord::from_message(message))
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
            response_meta: None,
        }]
        .into(),
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

fn find_progress(effects: &[Effect]) -> Option<(&MessageSequence, usize)> {
    effects.iter().find_map(|effect| match effect {
        Effect::Progress {
            messages,
            events: _,
            iteration,
        } => Some((messages, *iteration)),
        _ => None,
    })
}

fn find_done(effects: &[Effect]) -> Option<(&MessageSequence, usize)> {
    effects.iter().find_map(|effect| match effect {
        Effect::Done {
            messages,
            events: _,
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

fn find_execution_surface_sync(effects: &[Effect]) -> Option<(EffectId, bool)> {
    effects.iter().find_map(|effect| match effect {
        Effect::SyncExecutionSurface {
            id,
            update_machine_config,
        } => Some((*id, *update_machine_config)),
        _ => None,
    })
}

struct ProseDriver;

impl ProtocolDriverHandle for ProseDriver {
    fn prepare_iteration(&self, ctx: DriverContextView<'_>) -> Vec<DriverAction> {
        vec![DriverAction::StartLlm {
            request: ctx.project_llm_request(false),
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
            DriverAction::AppendEvents(vec![conversation_event(text_message(
                MessageRole::Assistant,
                "done",
            ))]),
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
            request: ctx.project_llm_request(false),
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
            DriverAction::AppendEvents(vec![conversation_event(text_message(
                MessageRole::Assistant,
                format!("state={marker}"),
            ))]),
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
            DriverAction::AppendEvents(vec![conversation_event(text_message(
                MessageRole::User,
                state,
            ))]),
            DriverAction::StartCheckpoint {
                checkpoint: CheckpointKind::BeforeCompletion,
                on_empty: CheckpointResumeAction::Finish,
            },
        ]
    }
}

struct SyncThenAdvanceDriver;

impl ProtocolDriverHandle for SyncThenAdvanceDriver {
    fn prepare_iteration(&self, ctx: DriverContextView<'_>) -> Vec<DriverAction> {
        vec![DriverAction::StartLlm {
            request: ctx.project_llm_request(true),
            driver_state: None,
        }]
    }

    fn handle_llm_success(
        &self,
        ctx: DriverContextView<'_>,
        _waiting: WaitingLlmState,
        _llm_response: LlmResponse,
        _text_streamed: bool,
    ) -> Vec<DriverAction> {
        if ctx.iteration() == ctx.run_offset() {
            vec![
                DriverAction::AdvanceIteration,
                DriverAction::StartCheckpoint {
                    checkpoint: CheckpointKind::BeforeCompletion,
                    on_empty: CheckpointResumeAction::PrepareIteration,
                },
            ]
        } else {
            vec![DriverAction::Finish]
        }
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
                    reference: test_attachment_ref(3),
                }),
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
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
                response_meta: None,
            },
        ]
        .into(),
        user_input: None,
        origin: None,
    }];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

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
            LlmContentBlock::Text { text, .. } => text.contains("explain this"),
            _ => false,
        })
    }));
}

#[test]
fn driver_can_finish_via_checkpoint() {
    let config = test_config(Arc::new(ProseDriver));
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call").0;
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "Hello".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Hello".to_string(),
                response_meta: None,
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
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

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
                response_meta: None,
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
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call").0;
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "Hello".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Hello".to_string(),
                response_meta: None,
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
    let (progress_messages, progress_iteration) =
        find_progress(&effects).expect("checkpoint progress");
    assert_eq!(progress_iteration, 1);
    assert!(progress_messages.iter().any(|message| {
        message.role == MessageRole::User
            && message
                .parts
                .iter()
                .any(|part| part.content == "one more thing")
    }));
    let progress_idx = effects
        .iter()
        .position(|effect| matches!(effect, Effect::Progress { .. }))
        .expect("progress index");
    let llm_idx = effects
        .iter()
        .position(|effect| matches!(effect, Effect::LlmCall { .. }))
        .expect("llm index");
    assert!(progress_idx < llm_idx);
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
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let (exec_id, code) = find_exec_call(&effects).expect("exec call");
    assert_eq!(code, "print 1");
    machine.handle_response(Response::ExecResult {
        id: *exec_id,
        result: Ok(crate::ExecResponse {
            output: String::new(),
            observations: Vec::new(),
            observation_truncation: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            printed_images: Vec::new(),
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

#[test]
fn initial_execution_surface_sync_is_host_only() {
    let mut config = test_config(Arc::new(ProseDriver));
    config.sync_execution_surface = true;
    let mut machine =
        TurnMachine::new(config, vec![user_message("hello")], Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let (sync_id, update_machine_config) =
        find_execution_surface_sync(&effects).expect("execution surface sync");
    assert!(!update_machine_config);

    machine.handle_response(Response::ExecutionSurfaceSynced {
        id: sync_id,
        result: Ok(None),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_llm_call(&effects).is_some());
}

#[test]
fn iteration_execution_surface_sync_can_refresh_prompt_and_tools() {
    let mut config = test_config(Arc::new(SyncThenAdvanceDriver));
    config.sync_execution_surface = true;
    config.system_prompt = Arc::from("initial prompt");
    let mut machine =
        TurnMachine::new(config, vec![user_message("hello")], Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let (initial_sync_id, update_machine_config) =
        find_execution_surface_sync(&effects).expect("initial execution surface sync");
    assert!(!update_machine_config);
    machine.handle_response(Response::ExecutionSurfaceSynced {
        id: initial_sync_id,
        result: Ok(None),
    });

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call").0;
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "advance".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "advance".to_string(),
                response_meta: None,
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
    let (sync_id, update_machine_config) =
        find_execution_surface_sync(&effects).expect("iteration execution surface sync");
    assert!(update_machine_config);

    machine.handle_response(Response::ExecutionSurfaceSynced {
        id: sync_id,
        result: Ok(Some(ExecutionSurfaceSync {
            system_prompt: Arc::from("updated prompt"),
            tool_specs: Arc::new(vec![crate::llm::types::LlmToolSpec {
                name: "new_tool".to_string(),
                description: "desc".to_string(),
                input_schema: serde_json::json!({ "type": "object" }),
                output_schema: serde_json::json!({ "type": "object" }),
            }]),
        })),
    });

    let effects = drain_effects(&mut machine);
    let (_, request) = find_llm_call(&effects).expect("second llm call");
    assert_eq!(request.tools.len(), 1);
    assert_eq!(request.tools[0].name, "new_tool");
    assert!(request.messages.iter().any(|message| {
        message.role == crate::llm::types::LlmRole::System
            && message.blocks.iter().any(|block| {
                matches!(
                    block,
                    crate::llm::types::LlmContentBlock::Text { text, .. }
                        if text.as_ref() == "updated prompt"
                )
            })
    }));
}
