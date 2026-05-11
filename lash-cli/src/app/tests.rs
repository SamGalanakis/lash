use super::*;
use crate::editor::LARGE_PASTE_CHAR_THRESHOLD;
use async_trait::async_trait;
use lash_core::{Part, PruneState, SessionEvent, TurnActivity, TurnEvent};
use lash_tui_extensions::{
    SlashCommandSpec, TuiExtension, TuiExtensionContext, TuiExtensions, TuiHostEffect,
};
use std::sync::Arc;

fn text_message(id: &str, role: MessageRole, content: &str) -> Message {
    Message {
        id: id.to_string(),
        role,
        parts: vec![part(&format!("{id}.p0"), PartKind::Text, content)].into(),
        origin: None,
    }
}

fn part(id: &str, kind: PartKind, content: &str) -> Part {
    Part {
        id: id.to_string(),
        kind,
        content: content.to_string(),
        attachment: None,
        tool_call_id: None,
        tool_name: None,
        tool_replay: None,
        prune_state: PruneState::Intact,
        reasoning_meta: None,
        response_meta: None,
    }
}

fn conversation_event(message: Message) -> lash_core::SessionEventRecord {
    lash_core::SessionEventRecord::Conversation(lash_core::ConversationRecord::from_message(
        message,
    ))
}

fn events_from_messages(messages: &[Message]) -> Vec<lash_core::SessionEventRecord> {
    messages.iter().cloned().map(conversation_event).collect()
}

fn test_read_view(
    events: &[lash_core::SessionEventRecord],
    messages: &[Message],
    tool_calls: &[ToolCallRecord],
) -> lash_core::SessionReadView {
    lash_core::SessionReadView::from_derived_message_view(
        lash_core::SessionStateEnvelope::default(),
        std::sync::Arc::new(events.to_vec()),
        std::sync::Arc::new(messages.to_vec()),
        std::sync::Arc::new(tool_calls.to_vec()),
    )
}

fn timeline_items_from_test_read_view(
    events: &[lash_core::SessionEventRecord],
    messages: &[Message],
    tool_calls: &[ToolCallRecord],
    ui_state: &UiProjectionState,
) -> Vec<UiTimelineItem> {
    let read_view = test_read_view(events, messages, tool_calls);
    timeline_from_read_view(&read_view, ui_state)
        .items()
        .to_vec()
}

fn timeline_from_test_read_view(
    events: &[lash_core::SessionEventRecord],
    messages: &[Message],
    tool_calls: &[ToolCallRecord],
    ui_state: &UiProjectionState,
) -> UiTimeline {
    let read_view = test_read_view(events, messages, tool_calls);
    timeline_from_read_view(&read_view, ui_state)
}

fn interrupted_blocks_from_test_read_view(
    events: &[lash_core::SessionEventRecord],
    messages: &[Message],
    tool_calls: &[ToolCallRecord],
    ui_state: &UiProjectionState,
    status_message: impl Into<String>,
) -> Vec<UiTimelineItem> {
    let read_view = test_read_view(events, messages, tool_calls);
    interrupted_blocks_from_read_view(&read_view, ui_state, status_message)
        .items()
        .to_vec()
}

fn other_variant_name(block: &UiTimelineItem) -> &'static str {
    match block {
        UiTimelineItem::TurnStart(_) => "TurnStart",
        UiTimelineItem::UserInput(_) => "UserInput",
        UiTimelineItem::AssistantText(_) => "AssistantText",
        UiTimelineItem::AssistantReasoning(_) => "AssistantReasoning",
        UiTimelineItem::Activity(_) => "Activity",
        UiTimelineItem::ShellOutput { .. } => "ShellOutput",
        UiTimelineItem::Error(_) => "Error",
        UiTimelineItem::SystemMessage(_) => "SystemMessage",
        UiTimelineItem::PluginPanel(_) => "PluginPanel",
        UiTimelineItem::LashlangCode(_) => "LashlangCode",
        UiTimelineItem::Splash => "Splash",
    }
}

mod projection;
mod render;
mod runtime_state;
