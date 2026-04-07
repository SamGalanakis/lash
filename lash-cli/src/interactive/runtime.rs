use std::collections::HashMap;
use std::sync::Arc;

use lash::session_model::{MessageRole, Part, PartKind, PruneState};
use lash::{PluginMessage, *};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::*;

/// Returned by the spawned runtime task so we can reclaim ownership.
pub(super) struct RuntimeRunResult {
    pub(super) stream_id: u64,
    pub(super) runtime: LashRuntime,
    pub(super) result: AssembledTurn,
}

pub(super) fn make_turn_input(
    _app: &mut App,
    items: Vec<InputItem>,
    image_blobs: HashMap<String, Vec<u8>>,
) -> TurnInput {
    TurnInput {
        items,
        image_blobs,
        mode: Some(RunMode::Normal),
    }
}

fn append_turn_input_message(messages: &mut Vec<Message>, turn_input: &TurnInput) {
    let user_id = format!("m{}", messages.len());
    let mut image_ids = Vec::new();
    let mut user_parts = Vec::new();

    for item in &turn_input.items {
        match item {
            InputItem::Text { text } => {
                if text.is_empty() {
                    continue;
                }
                user_parts.push(Part {
                    id: format!("{}.p{}", user_id, user_parts.len()),
                    kind: PartKind::Text,
                    content: text.clone(),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
            InputItem::FileRef { path } => {
                user_parts.push(Part {
                    id: format!("{}.p{}", user_id, user_parts.len()),
                    kind: PartKind::Text,
                    content: format!("[file: {path}]"),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
            InputItem::DirRef { path } => {
                user_parts.push(Part {
                    id: format!("{}.p{}", user_id, user_parts.len()),
                    kind: PartKind::Text,
                    content: format!("[directory: {}]", path.trim_end_matches('/')),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
            InputItem::ImageRef { id } => {
                let Some(bytes) = turn_input.image_blobs.get(id) else {
                    continue;
                };
                if image_ids.iter().any(|candidate| candidate == id) {
                    continue;
                }
                image_ids.push(id.clone());
                user_parts.push(Part {
                    id: format!("{}.p{}", user_id, user_parts.len()),
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

    if user_parts.is_empty() {
        user_parts.push(Part {
            id: format!("{user_id}.p0"),
            kind: PartKind::Text,
            content: String::new(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        });
    }

    messages.push(Message {
        id: user_id,
        role: MessageRole::User,
        parts: user_parts,
        origin: None,
    });
}

fn pending_turn_snapshot(
    state: &SessionStateEnvelope,
    turn_input: &TurnInput,
) -> DurableTurnSnapshot {
    let mut messages = state.messages.clone();
    append_turn_input_message(&mut messages, turn_input);
    DurableTurnSnapshot {
        messages,
        tool_calls: state.tool_calls.clone(),
        iteration: state.iteration,
    }
}

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
    provider: &Provider,
    dynamic_state: &DynamicStateSnapshot,
    toolset_hash: &str,
) {
    let mut ui_trace = ui_trace;
    let already_visible = if !prepared_turn.display_text.is_empty() {
        app.commit_pending_user_preview(&prepared_turn.display_text)
    } else {
        false
    };
    if !prepared_turn.display_text.is_empty() && !already_visible {
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

    let mut rt = runtime
        .take()
        .expect("runtime should be available when not running");
    let persisted_state = rt.export_state();
    persist_live_runtime_snapshot(
        logger.store().as_ref(),
        pending_turn_snapshot(&persisted_state, &turn_input),
        &app.ui_resume_state(),
        dynamic_state,
        provider,
        &app.model,
        app.context_window
            .expect("app context_window must be set before dispatching a turn"),
        persisted_state.policy.execution_mode,
        persisted_state.policy.context_strategy,
        app.model_variant.as_deref(),
        toolset_hash,
        persisted_state.token_usage.clone(),
        persisted_state.last_prompt_usage.clone(),
    );
    tracing::info!(
        mode = ?turn_input.mode,
        items = turn_input.items.len(),
        images = turn_input.image_blobs.len(),
        "dispatching runtime turn"
    );
    let (return_tx, return_rx) = tokio::sync::oneshot::channel();
    *runtime_return_rx = Some(return_rx);

    let cancel = CancellationToken::new();
    *cancel_token = Some(cancel.clone());
    *active_stream_id = active_stream_id.wrapping_add(1);
    let stream_id = *active_stream_id;

    let sink_tx = app_tx.clone();
    tokio::spawn(async move {
        let sink = AppEventSink {
            tx: sink_tx,
            stream_id,
        };
        let result = match rt.stream_turn(turn_input, &sink, cancel).await {
            Ok(turn) => turn,
            Err(e) => AssembledTurn {
                state: rt.export_state(),
                status: TurnStatus::Failed,
                assistant_output: AssistantOutput {
                    safe_text: String::new(),
                    raw_text: String::new(),
                    state: OutputState::EmptyOutput,
                },
                has_plugin_visible_output: false,
                done_reason: DoneReason::RuntimeError,
                execution: ExecutionSummary {
                    mode: rt.export_state().policy.execution_mode,
                    had_tool_calls: false,
                    had_code_execution: false,
                },
                token_usage: TokenUsage::default(),
                tool_calls: Vec::new(),
                code_outputs: Vec::new(),
                errors: vec![TurnIssue {
                    kind: "runtime".to_string(),
                    code: Some(e.code),
                    message: e.message,
                }],
            },
        };
        let _ = return_tx.send(RuntimeRunResult {
            stream_id,
            runtime: rt,
            result,
        });
    });
}

/// Send a desktop notification that the session finished.
pub(super) fn notify_done() {
    let icon_path = lash::lash_home().join("icon.svg");
    if !icon_path.exists() {
        let _ = std::fs::write(&icon_path, include_bytes!("../../assets/icon.svg"));
    }
    let _ = std::process::Command::new("notify-send")
        .args(["-a", "lash", "-i"])
        .arg(&icon_path)
        .args(["lash", "Response complete"])
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

/// Copy the current history selection when present, otherwise fall back to the
/// last assistant response.
pub(super) fn copy_selected_text_or_last_response(app: &App, terminal_size: Option<(u16, u16)>) {
    let selected_text = terminal_size.and_then(|(width, height)| {
        crate::render::extract_history_selection_text(app, width, height)
    });
    let last_text = app.blocks.iter().rev().find_map(|b| {
        if let DisplayBlock::AssistantText(text) = b {
            Some(text.clone())
        } else {
            None
        }
    });
    if let Some(text) = selected_text.or(last_text)
        && let Ok(mut clipboard) = arboard::Clipboard::new()
    {
        let _ = clipboard.set_text(text);
    }
}
