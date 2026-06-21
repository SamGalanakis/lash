use lash::persistence::FileAttachmentStore;
use lash::{InputItem, LashSession, TurnInput};
use lash_core::session_model::{Part, PartKind, PruneState};
use lash_core::{
    AttachmentCreateMeta, AttachmentStore, ImageMediaType, MediaType, Message, MessageRole,
    PluginMessage,
};

use super::helpers::SessionObservationBridge;
use super::*;
use crate::app::ToastKind;
use crate::event::AppEventTx;
use crate::input_items::build_items_from_editor_input;
use crate::turn_runner::{
    RuntimeRunResult, make_turn_input, spawn_session_queued_turn, spawn_session_turn,
};

pub(crate) async fn make_injected_plugin_message(turn: &PreparedTurn) -> PluginMessage {
    let (items, image_blobs) =
        build_items_from_editor_input(&turn.effective_text, turn.images.clone());
    let attachment_store = FileAttachmentStore::new(crate::paths::attachments_dir());
    let mut parts = Vec::new();
    let mut image_ids = Vec::new();
    for item in items {
        match item {
            InputItem::Text { text } => {
                if text.is_empty() {
                    continue;
                }
                parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Text,
                    content: text,
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_replay: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                });
            }
            InputItem::ImageRef { id } => {
                let Some(bytes) = image_blobs.get(&id) else {
                    continue;
                };
                if image_ids.iter().any(|candidate| candidate == &id) {
                    continue;
                }
                image_ids.push(id.clone());
                let meta = AttachmentCreateMeta::new(
                    MediaType::Image(ImageMediaType::Png),
                    None,
                    None,
                    Some(id.clone()),
                );
                let Ok(reference) = attachment_store.put(bytes.clone(), meta).await else {
                    continue;
                };
                parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Image,
                    content: String::new(),
                    attachment: Some(lash_core::session_model::message::PartAttachment {
                        reference,
                    }),
                    tool_call_id: None,
                    tool_name: None,
                    tool_replay: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                });
            }
        }
    }

    PluginMessage {
        role: MessageRole::User,
        content: turn.effective_text.clone(),
        origin: None,
        parts,
        images: Vec::new(),
    }
}

#[cfg(test)]
pub(crate) fn injected_image_part_indices(message: &PluginMessage) -> Vec<usize> {
    message
        .parts
        .iter()
        .enumerate()
        .filter_map(|(idx, part)| matches!(part.kind, PartKind::Image).then_some(idx))
        .collect()
}

pub(super) async fn sync_runtime_tool_catalog(
    runtime: &mut Option<LashSession>,
) -> Result<(), String> {
    if let Some(rt) = runtime.as_mut() {
        rt.admin()
            .commands()
            .refresh_tool_catalog("interactive sync", "interactive-sync-runtime-tools")
            .await
            .map_err(|err| err.to_string())?;
    }
    Ok(())
}

pub(super) async fn enqueue_prepared_turn(
    session: &LashSession,
    turn: &PreparedTurn,
    delivery_policy: lash_core::DeliveryPolicy,
    slot_policy: lash_core::SlotPolicy,
) -> Result<(), String> {
    session
        .enqueue(make_turn_input(turn))
        .id(turn.draft_id.clone())
        .delivery_policy(delivery_policy)
        .slot_policy(slot_policy)
        .send()
        .await
        .map(|_| ())
        .map_err(|err| err.to_string())
}

pub(super) async fn refresh_queued_work_snapshot(
    app: &mut App,
    runtime: &Option<LashSession>,
) -> Result<(), String> {
    let Some(session) = runtime.as_ref() else {
        app.clear_queued_work_snapshot();
        return Ok(());
    };
    let queued = session.queued_work().await.map_err(|err| err.to_string())?;
    app.set_queued_work_snapshot(queued);
    Ok(())
}

/// Send a user message to the runtime: push display block and spawn turn run.
#[allow(clippy::too_many_arguments)]
pub(super) async fn send_user_message(
    prepared_turn: PreparedTurn,
    turn_input: TurnInput,
    app: &mut App,
    ui_trace: Option<&mut UiTraceRecorder>,
    logger: &mut SessionLogger,
    runtime: &mut Option<LashSession>,
    _history: &mut Vec<Message>,
    runtime_return_rx: &mut Option<tokio::sync::oneshot::Receiver<RuntimeRunResult>>,
    cancel_token: &mut Option<CancellationToken>,
    active_stream_id: &mut u64,
    app_tx: &AppEventTx,
) {
    let mut ui_trace = ui_trace;
    if !prepared_turn.display_text.is_empty() {
        if let Some(recorder) = ui_trace.as_deref_mut() {
            recorder.record_user_turn(&prepared_turn);
        }
        let _ = logger.record_host_input(&prepared_turn);
        app.push_prepared_user_input(&prepared_turn);
    }
    if let Some(recorder) = ui_trace {
        recorder.record_start_turn();
    }
    app.start_turn();
    app.mark_live_turn_user_input_visible();
    app.resume_contextual_follow_output();
    app.keep_latest_user_block_visible();

    tracing::debug!(
        display_text = prepared_turn.display_text,
        runtime_present_before_take = runtime.is_some(),
        runtime_return_rx_present_before_take = runtime_return_rx.is_some(),
        cancel_token_present_before_take = cancel_token.is_some(),
        queued_work = app.queued_work_snapshot().len(),
        draft_presentations = app.queues.draft_presentations.len(),
        "send_user_message taking runtime for dispatch"
    );

    let session = runtime
        .as_ref()
        .cloned()
        .expect("runtime should be available when not running");
    tracing::info!(
        items = turn_input.items.len(),
        images = turn_input.image_blobs.len(),
        "dispatching runtime turn"
    );
    *active_stream_id = active_stream_id.wrapping_add(1);
    let stream_id = *active_stream_id;

    tracing::debug!(
        stream_id,
        runtime_present_after_take = runtime.is_some(),
        runtime_return_rx_present_after_set = runtime_return_rx.is_some(),
        cancel_token_present_after_set = cancel_token.is_some(),
        "send_user_message armed runtime return channel"
    );

    SessionObservationBridge::spawn(&session, stream_id, app_tx.clone());
    let (cancel, return_rx) = spawn_session_turn(session, turn_input, stream_id);
    *cancel_token = Some(cancel);
    *runtime_return_rx = Some(return_rx);
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn send_queued_work(
    batch_ids: Vec<String>,
    prepared_turns: Vec<PreparedTurn>,
    app: &mut App,
    ui_trace: Option<&mut UiTraceRecorder>,
    logger: &mut SessionLogger,
    runtime: &mut Option<LashSession>,
    runtime_return_rx: &mut Option<tokio::sync::oneshot::Receiver<RuntimeRunResult>>,
    cancel_token: &mut Option<CancellationToken>,
    active_stream_id: &mut u64,
    app_tx: &AppEventTx,
) {
    let mut ui_trace = ui_trace;
    for prepared_turn in &prepared_turns {
        if prepared_turn.display_text.is_empty() {
            continue;
        }
        if let Some(recorder) = ui_trace.as_deref_mut() {
            recorder.record_user_turn(prepared_turn);
        }
        let _ = logger.record_host_input(prepared_turn);
        app.push_prepared_user_input(prepared_turn);
    }
    if let Some(recorder) = ui_trace {
        recorder.record_start_turn();
    }
    let has_visible_user_input = prepared_turns
        .iter()
        .any(|turn| !turn.display_text.trim().is_empty());
    app.start_turn();
    if has_visible_user_input {
        app.mark_live_turn_user_input_visible();
    }
    app.resume_contextual_follow_output();
    app.keep_latest_user_block_visible();

    let session = runtime
        .as_ref()
        .cloned()
        .expect("runtime should be available when dispatching queued work");
    *active_stream_id = active_stream_id.wrapping_add(1);
    let stream_id = *active_stream_id;
    tracing::info!(
        stream_id,
        prepared_turns = prepared_turns.len(),
        "dispatching durable queued runtime turn"
    );

    SessionObservationBridge::spawn(&session, stream_id, app_tx.clone());
    let (cancel, return_rx) = spawn_session_queued_turn(session, batch_ids, stream_id);
    *cancel_token = Some(cancel);
    *runtime_return_rx = Some(return_rx);
}

pub(crate) fn notify_desktop(title: &str, body: &str) {
    let icon_path = crate::paths::lash_home().join("icon.svg");
    if !icon_path.exists() {
        let _ = std::fs::write(&icon_path, include_bytes!("../../assets/icon.svg"));
    }
    let _ = std::process::Command::new("notify-send")
        .args(["-a", "lash", "-i"])
        .arg(&icon_path)
        .arg(title)
        .arg(body)
        .spawn();
}

/// Generate a unique session name like "juniper-mountain".
/// Scans existing session files for collisions.
pub(crate) async fn generate_session_name(sessions_dir: &std::path::Path) -> String {
    use rand::Rng;

    const ADJECTIVES: &[&str] = &[
        "alpine", "amber", "ancient", "ashen", "autumn", "blazing", "bright", "calm", "cedar",
        "coastal", "copper", "coral", "crimson", "crystal", "dappled", "deep", "desert", "distant",
        "dusky", "ember", "fading", "fern", "flint", "foggy", "forest", "frozen", "gentle",
        "gilded", "glacial", "golden", "granite", "hollow", "iron", "ivory", "jade", "keen",
        "lofty", "lunar", "marble", "misty", "mossy", "northern", "obsidian", "onyx", "opal",
        "pale", "pine", "quiet", "radiant", "rugged", "rustic", "sandy", "silver", "silent",
        "solar", "stone", "sunlit", "tidal", "twilight", "verdant", "violet", "wild", "winter",
    ];
    const NOUNS: &[&str] = &[
        "basin",
        "birch",
        "bluff",
        "boulder",
        "brook",
        "canyon",
        "cavern",
        "cliff",
        "cove",
        "creek",
        "delta",
        "dune",
        "falls",
        "field",
        "fjord",
        "glade",
        "gorge",
        "grove",
        "harbor",
        "heath",
        "hill",
        "island",
        "lake",
        "ledge",
        "marsh",
        "meadow",
        "mesa",
        "mountain",
        "oasis",
        "ocean",
        "pass",
        "peak",
        "plain",
        "plateau",
        "pond",
        "prairie",
        "ravine",
        "reef",
        "ridge",
        "river",
        "shore",
        "slope",
        "spring",
        "stone",
        "summit",
        "terrace",
        "thicket",
        "timber",
        "trail",
        "tundra",
        "vale",
        "valley",
        "vista",
        "volcano",
        "waterfall",
        "willow",
        "woods",
    ];

    let mut existing = std::collections::HashSet::new();
    if let Ok(entries) = std::fs::read_dir(sessions_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("db")
                && let Some(filename) = path.file_name().and_then(|name| name.to_str())
                && let Ok(start) = session_log::load_session_start(filename).await
            {
                existing.insert(start.session_name);
            }
        }
    }

    let mut rng = rand::rng();
    loop {
        let adj = ADJECTIVES[rng.random_range(0..ADJECTIVES.len())];
        let noun = NOUNS[rng.random_range(0..NOUNS.len())];
        let name = format!("{adj}-{noun}");
        if !existing.contains(&name) {
            return name;
        }
    }
}

/// Copy the current text selection only. Used by mouse-up copy-on-select.
pub(super) fn copy_active_selection(app: &mut App, terminal_size: Option<(u16, u16)>) -> bool {
    let input_text = app.selected_input_text();
    let document_text = document_selection_text(app, terminal_size);
    let history_text = history_selection_text(app, terminal_size);
    tracing::debug!(
        selection_visible = app.selection.visible,
        selection_active = app.selection.active,
        input_selected_chars = input_text.as_ref().map(|text| text.chars().count()),
        document_selected_chars = document_text.as_ref().map(|text| text.chars().count()),
        history_selected_chars = history_text.as_ref().map(|text| text.chars().count()),
        has_terminal_size = terminal_size.is_some(),
        "selection copy path invoked"
    );
    let text = input_text.or(document_text).or(history_text);
    let Some(text) = text else {
        return false;
    };
    copy_text_with_toast(app, &text);
    app.clear_text_selection();
    true
}

/// Copy the current input selection when present, otherwise the history
/// selection, and finally the last assistant response.
pub(super) fn copy_selected_text_or_last_response(
    app: &mut App,
    terminal_size: Option<(u16, u16)>,
) {
    let input_text = app.selected_input_text();
    let document_text = document_selection_text(app, terminal_size);
    let history_text = history_selection_text(app, terminal_size);
    let last_text = last_assistant_response_text(app);
    tracing::debug!(
        selection_visible = app.selection.visible,
        selection_active = app.selection.active,
        input_selected_chars = input_text.as_ref().map(|text| text.chars().count()),
        document_selected_chars = document_text.as_ref().map(|text| text.chars().count()),
        history_selected_chars = history_text.as_ref().map(|text| text.chars().count()),
        fallback_chars = last_text.as_ref().map(|text| text.chars().count()),
        has_terminal_size = terminal_size.is_some(),
        "copy path invoked"
    );
    if let Some(text) = copy_candidate_text(app, terminal_size) {
        copy_text_with_toast(app, &text);
    } else {
        tracing::debug!("copy path had no selected or fallback text");
    }
}

fn copy_text_with_toast(app: &mut App, text: &str) {
    let copied_chars = text.chars().count();
    match crate::clipboard::copy_text_robustly(text) {
        Ok(method) => {
            tracing::debug!(copied_chars, method, "clipboard write succeeded");
            app.show_toast("Copied to clipboard", ToastKind::Info);
        }
        Err(err) => {
            tracing::warn!(error = %err, copied_chars, "clipboard write failed");
            app.show_toast("Failed to copy to clipboard", ToastKind::Error);
        }
    }
}

fn copy_candidate_text(app: &App, terminal_size: Option<(u16, u16)>) -> Option<String> {
    app.selected_input_text()
        .or_else(|| document_selection_text(app, terminal_size))
        .or_else(|| history_selection_text(app, terminal_size))
        .or_else(|| last_assistant_response_text(app))
}

fn document_selection_text(app: &App, terminal_size: Option<(u16, u16)>) -> Option<String> {
    terminal_size.and_then(|(width, height)| {
        crate::render::extract_document_selection_text(app, width, height)
    })
}

fn history_selection_text(app: &App, terminal_size: Option<(u16, u16)>) -> Option<String> {
    terminal_size.and_then(|(width, height)| {
        crate::render::extract_history_selection_text(app, width, height)
    })
}

fn last_assistant_response_text(app: &App) -> Option<String> {
    app.timeline.iter().rev().find_map(|block| {
        if let UiTimelineItem::AssistantText(text) = block {
            Some(text.clone())
        } else {
            None
        }
    })
}

#[cfg(test)]
mod copy_tests {
    use super::*;

    #[test]
    fn copy_candidate_prefers_input_selection_over_history_and_fallback() {
        let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
        app.timeline = vec![
            UiTimelineItem::UserInput("history alpha".into()),
            UiTimelineItem::AssistantText("assistant fallback".into()),
        ]
        .into();
        app.selection.anchor = (2, 0);
        app.selection.end = (9, 0);
        app.selection.visible = true;
        app.set_input("input beta".into());
        app.start_input_selection(0);
        app.update_input_selection(5);
        app.finish_input_selection();

        assert_eq!(
            copy_candidate_text(&app, Some((80, 24))).as_deref(),
            Some("input")
        );
    }

    #[test]
    fn copy_candidate_uses_history_selection_before_fallback() {
        let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
        app.timeline = vec![
            UiTimelineItem::UserInput("history alpha".into()),
            UiTimelineItem::AssistantText("assistant fallback".into()),
        ]
        .into();
        app.selection.anchor = (2, 0);
        app.selection.end = (9, 0);
        app.selection.visible = true;

        assert_eq!(
            copy_candidate_text(&app, Some((80, 24))).as_deref(),
            Some("history")
        );
    }

    #[test]
    fn copy_candidate_falls_back_to_last_assistant_response_without_selection() {
        let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
        app.timeline = vec![
            UiTimelineItem::AssistantText("older".into()),
            UiTimelineItem::UserInput("question".into()),
            UiTimelineItem::AssistantText("newer".into()),
        ]
        .into();

        assert_eq!(
            copy_candidate_text(&app, Some((80, 24))).as_deref(),
            Some("newer")
        );
    }
}
