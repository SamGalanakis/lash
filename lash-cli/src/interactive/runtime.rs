use std::collections::HashMap;
use std::sync::Arc;

use lash::session_model::{Part, PartKind, PruneState};
use lash::{PluginMessage, *};

use super::*;
use crate::input_items::build_items_from_editor_input;
#[cfg(test)]
use crate::turn_runner::pending_turn_snapshot;
use crate::turn_runner::{RuntimeRunResult, persist_pending_turn, spawn_runtime_turn};

pub(crate) fn make_injected_plugin_message(turn: &PreparedTurn) -> PluginMessage {
    let (items, image_blobs) =
        build_items_from_editor_input(&turn.effective_text, turn.images.clone());
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
                    prune_state: PruneState::Intact,
                });
            }
            InputItem::FileRef { path } => {
                parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Text,
                    content: format!("[file: {path}]"),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
            InputItem::DirRef { path } => {
                parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Text,
                    content: format!("[directory: {}]", path.trim_end_matches('/')),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
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
                parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Image,
                    content: String::new(),
                    attachment: Some(lash::session_model::message::PartAttachment {
                        mime: "image/png".to_string(),
                        url: lash::session_model::message::data_url_for_bytes("image/png", bytes),
                        filename: None,
                    }),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
        }
    }

    PluginMessage {
        role: MessageRole::User,
        content: turn.effective_text.clone(),
        parts,
        images: Vec::new(),
        user_input: Some(turn.input_provenance.clone()),
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

pub(super) fn parse_kv_args(raw: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for token in raw.split_whitespace() {
        if let Some((k, v)) = token.split_once('=') {
            out.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    out
}

pub(super) fn register_builtin_tool(
    dynamic_tools: &Arc<DynamicToolProvider>,
    tool_name: &str,
    handler_id: &str,
    description_override: Option<String>,
    _execution_mode: ExecutionMode,
) -> Result<ToolDefinition, String> {
    let adapter = dynamic_tools.inprocess_adapter();
    let def = match handler_id {
        "echo" => {
            let handler: InProcessToolHandler = Arc::new(|args, _context, _progress| {
                Box::pin(async move {
                    let text = args
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    ToolResult::ok(serde_json::json!(text))
                })
            });
            let def = ToolDefinition {
                name: tool_name.to_string(),
                description: description_override
                    .unwrap_or_else(|| "Echoes back the `text` argument.".to_string()),
                params: vec![ToolParam::typed("text", "str")],
                returns: "str".to_string(),
                examples: vec![format!("{tool_name}(text=\"hello\")")],
                enabled: true,
                injected: false,
                input_schema_override: None,
                output_schema_override: None,
            };
            adapter.register_tool(def.clone(), handler);
            def
        }
        "time" => {
            let handler: InProcessToolHandler = Arc::new(|_args, _context, _progress| {
                Box::pin(async move {
                    ToolResult::ok(serde_json::json!(chrono::Utc::now().to_rfc3339()))
                })
            });
            let def = ToolDefinition {
                name: tool_name.to_string(),
                description: description_override
                    .unwrap_or_else(|| "Returns the current UTC timestamp (RFC3339).".to_string()),
                params: vec![],
                returns: "str".to_string(),
                examples: vec![format!("{tool_name}()")],
                enabled: true,
                injected: false,
                input_schema_override: None,
                output_schema_override: None,
            };
            adapter.register_tool(def.clone(), handler);
            def
        }
        "uuid" => {
            let handler: InProcessToolHandler = Arc::new(|_args, _context, _progress| {
                Box::pin(async move {
                    ToolResult::ok(serde_json::json!(uuid::Uuid::new_v4().to_string()))
                })
            });
            let def = ToolDefinition {
                name: tool_name.to_string(),
                description: description_override
                    .unwrap_or_else(|| "Returns a random UUIDv4 string.".to_string()),
                params: vec![],
                returns: "str".to_string(),
                examples: vec![format!("{tool_name}()")],
                enabled: true,
                injected: false,
                input_schema_override: None,
                output_schema_override: None,
            };
            adapter.register_tool(def.clone(), handler);
            def
        }
        other => {
            return Err(format!(
                "Unknown handler `{other}`. Supported handlers: echo, time, uuid"
            ));
        }
    };

    Ok(def)
}

pub(super) async fn apply_pending_reconfigure(
    dynamic_tools: &Arc<DynamicToolProvider>,
    desired_dynamic: &mut DynamicStateSnapshot,
    pending_reconfigure: &mut bool,
    runtime: &mut Option<LashRuntime>,
) -> Result<u64, String> {
    if !*pending_reconfigure {
        return Ok(dynamic_tools.generation());
    }

    let previous = dynamic_tools.export_state();
    let generation = match dynamic_tools.apply_state(desired_dynamic.clone()) {
        Ok(g) => g,
        Err(e) => {
            desired_dynamic.base_generation = dynamic_tools.generation();
            return Err(e.to_string());
        }
    };

    if let Some(rt) = runtime.as_mut()
        && let Err(e) = rt.refresh_session_execution_surface().await
    {
        let mut rollback = previous.clone();
        rollback.base_generation = dynamic_tools.generation();
        let _ = dynamic_tools.apply_state(rollback);
        let _ = rt.refresh_session_execution_surface().await;
        desired_dynamic.base_generation = dynamic_tools.generation();
        return Err(format!(
            "Failed to apply runtime reconfigure (state rolled back): {e}"
        ));
    }

    *desired_dynamic = dynamic_tools.export_state();
    *pending_reconfigure = false;
    Ok(generation)
}

/// Send a user message to the runtime: push display block and spawn turn run.
#[allow(clippy::too_many_arguments)]
pub(super) fn send_user_message(
    prepared_turn: PreparedTurn,
    turn_input: TurnInput,
    app: &mut App,
    ui_trace: Option<&mut UiTraceRecorder>,
    logger: &mut SessionLogger,
    runtime: &mut Option<LashRuntime>,
    _history: &mut Vec<Message>,
    runtime_return_rx: &mut Option<tokio::sync::oneshot::Receiver<RuntimeRunResult>>,
    cancel_token: &mut Option<CancellationToken>,
    active_stream_id: &mut u64,
    app_tx: &mpsc::UnboundedSender<AppEvent>,
    dynamic_state: &DynamicStateSnapshot,
) {
    let mut ui_trace = ui_trace;
    if !prepared_turn.display_text.is_empty() {
        if let Some(recorder) = ui_trace.as_deref_mut() {
            recorder.record_user_turn(&prepared_turn);
        }
        app.push_prepared_user_input(&prepared_turn);
    }
    if let Some(recorder) = ui_trace {
        recorder.record_start_turn();
    }
    app.start_turn();
    app.resume_contextual_follow_output();
    app.keep_latest_user_block_visible();

    tracing::debug!(
        display_text = prepared_turn.display_text,
        runtime_present_before_take = runtime.is_some(),
        runtime_return_rx_present_before_take = runtime_return_rx.is_some(),
        cancel_token_present_before_take = cancel_token.is_some(),
        queued_turns = app.queued_turns.len(),
        pending_steers = app.pending_steers.len(),
        "send_user_message taking runtime for dispatch"
    );

    let rt = runtime
        .take()
        .expect("runtime should be available when not running");
    persist_pending_turn(
        logger.store().as_ref(),
        &rt,
        &turn_input,
        &app.ui_resume_state(),
        dynamic_state,
    );
    tracing::info!(
        mode = ?turn_input.mode,
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

    let sink_tx = app_tx.clone();
    let sink = AppEventSink {
        tx: sink_tx,
        stream_id,
    };
    let (cancel, return_rx) = spawn_runtime_turn(rt, turn_input, sink, stream_id);
    *cancel_token = Some(cancel);
    *runtime_return_rx = Some(return_rx);
}

pub(crate) fn notify_desktop(title: &str, body: &str) {
    let icon_path = lash::lash_home().join("icon.svg");
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
pub(crate) fn generate_session_name(sessions_dir: &std::path::Path) -> String {
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
                && let Ok(start) = session_log::load_session_start(filename)
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

/// Copy the current input selection when present, otherwise the history
/// selection, and finally the last assistant response.
pub(super) fn copy_selected_text_or_last_response(app: &App, terminal_size: Option<(u16, u16)>) {
    let input_text = app.selected_input_text();
    let history_text = terminal_size.and_then(|(width, height)| {
        crate::render::extract_history_selection_text(app, width, height)
    });
    let last_text = app.blocks.iter().rev().find_map(|b| {
        if let DisplayBlock::AssistantText(text) = b {
            Some(text.clone())
        } else {
            None
        }
    });
    tracing::debug!(
        selection_visible = app.selection.visible,
        selection_active = app.selection.active,
        input_selected_chars = input_text.as_ref().map(|text| text.chars().count()),
        history_selected_chars = history_text.as_ref().map(|text| text.chars().count()),
        fallback_chars = last_text.as_ref().map(|text| text.chars().count()),
        has_terminal_size = terminal_size.is_some(),
        "copy path invoked"
    );
    if let Some(text) = input_text.or(history_text).or(last_text) {
        let copied_chars = text.chars().count();
        match crate::clipboard::copy_text_robustly(&text) {
            Ok(method) => tracing::debug!(copied_chars, method, "clipboard write succeeded"),
            Err(err) => tracing::warn!(error = %err, copied_chars, "clipboard write failed"),
        }
    } else {
        tracing::debug!("copy path had no selected or fallback text");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash::{SkillCatalog, UserInputProvenance};

    #[test]
    fn pending_turn_snapshot_preserves_user_input_provenance() {
        let turn = PreparedTurn::prepare("/yolopush".into(), Vec::new(), &SkillCatalog::default());
        let turn_input = crate::turn_runner::make_turn_input(&turn);
        let snapshot = pending_turn_snapshot(&SessionStateEnvelope::default(), &turn_input);

        let user_message = snapshot.messages.last().expect("user message");
        assert_eq!(
            user_message
                .user_input
                .as_ref()
                .map(|input| input.display_text.as_str()),
            Some("/yolopush")
        );
    }

    #[test]
    fn pending_turn_snapshot_keeps_explicit_user_input_provenance() {
        let turn_input = TurnInput {
            items: vec![InputItem::Text {
                text: "/localref\n\n<skill>\nbody\n</skill>".into(),
            }],
            image_blobs: HashMap::new(),
            user_input: Some(UserInputProvenance {
                display_text: "/localref".into(),
                effective_text: "/localref\n\n<skill>\nbody\n</skill>".into(),
                transforms: vec![lash::UserInputTransform::SkillBlockAppend {
                    skill_name: "localref".into(),
                    skill_path: "/tmp/localref/SKILL.md".into(),
                }],
            }),
            mode: Some(RunMode::Normal),
        };
        let snapshot = pending_turn_snapshot(&SessionStateEnvelope::default(), &turn_input);

        let user_message = snapshot.messages.last().expect("user message");
        assert_eq!(
            user_message
                .user_input
                .as_ref()
                .map(|input| input.display_text.as_str()),
            Some("/localref")
        );
    }
}
