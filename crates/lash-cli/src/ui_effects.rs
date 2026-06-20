//! Applying plugin/UI-extension host effects to the TUI [`App`], plus the
//! shared system-message and UI-snapshot helpers.

use lash::{LashSession, TurnEvent, TurnInput};
use lash_core::{PluginOwned, PluginRuntimeEvent};
use lash_tui_extensions::{TuiExtensions, TuiHostEffect};

use crate::app::{App, PluginPanelBlock, UiTimelineItem};

pub(crate) fn push_system_message(app: &mut App, msg: impl Into<String>) {
    let msg = msg.into();
    let duplicate = matches!(
        app.timeline.last(),
        Some(UiTimelineItem::SystemMessage(existing)) if existing == &msg
    );
    if duplicate {
        return;
    }
    crate::ui_trace::record_system_message_aux(&msg);
    app.timeline.push(UiTimelineItem::SystemMessage(msg));
    app.invalidate_height_cache();
    app.scroll_to_bottom();
}

pub(crate) fn apply_ui_host_effects(app: &mut App, effects: Vec<TuiHostEffect>) {
    for effect in effects {
        match effect {
            TuiHostEffect::PushSystemMessage(message) => push_system_message(app, message),
            TuiHostEffect::DesktopNotification {
                title,
                body,
                only_when_unfocused,
            } => {
                if !only_when_unfocused || !app.focused {
                    crate::interactive::notify_desktop(&title, &body);
                }
            }
            TuiHostEffect::UpsertModeIndicator { key, label } => {
                app.upsert_mode_indicator(key, label);
            }
            TuiHostEffect::ClearModeIndicator { key } => {
                app.clear_mode_indicator(&key);
            }
            TuiHostEffect::UpsertPanel {
                plugin_id,
                key,
                title,
                content,
            } => {
                let target = crate::plugin_surface::surface_key(&plugin_id, &key);
                if let Some(existing) = app.timeline.iter_mut().find_map(|block| match block {
                    UiTimelineItem::PluginPanel(panel)
                        if crate::plugin_surface::surface_key(&panel.plugin_id, &panel.key)
                            == target =>
                    {
                        Some(panel)
                    }
                    _ => None,
                }) {
                    existing.title = title;
                    existing.content = content;
                } else {
                    app.timeline
                        .push(UiTimelineItem::PluginPanel(PluginPanelBlock {
                            plugin_id,
                            key,
                            title,
                            content,
                        }));
                }
                app.invalidate_height_cache();
                app.scroll_to_bottom();
                app.dirty = true;
            }
            TuiHostEffect::ClearPanel { plugin_id, key } => {
                let target = crate::plugin_surface::surface_key(&plugin_id, &key);
                let original_len = app.timeline.len();
                app.timeline.retain(|block| match block {
                    UiTimelineItem::PluginPanel(panel) => {
                        crate::plugin_surface::surface_key(&panel.plugin_id, &panel.key) != target
                    }
                    _ => true,
                });
                if app.timeline.len() != original_len {
                    app.invalidate_height_cache();
                    app.dirty = true;
                }
            }
            TuiHostEffect::QueueTurn { input } => {
                let _ = input;
                push_system_message(
                    app,
                    "Queue requests from UI surfaces require a live runtime queue.".to_string(),
                );
            }
            TuiHostEffect::QueuePreparedTurn {
                display_text,
                effective_text,
            } => {
                let _ = (display_text, effective_text);
                push_system_message(
                    app,
                    "Queue requests from UI surfaces require a live runtime queue.".to_string(),
                );
            }
            TuiHostEffect::RunPluginCommand { name, args }
            | TuiHostEffect::RunPluginTask { name, args } => {
                let _ = (name, args);
                push_system_message(
                    app,
                    "Plugin operation requests require a live runtime session.".to_string(),
                );
            }
            TuiHostEffect::MountSurface { .. }
            | TuiHostEffect::UpdateSurface { .. }
            | TuiHostEffect::UnmountSurface { .. }
            | TuiHostEffect::FocusSurface { .. }
            | TuiHostEffect::BlurSurface { .. } => {
                app.dirty = true;
            }
        }
    }
}

pub(crate) async fn apply_ui_host_effects_with_runtime(
    app: &mut App,
    ui_extensions: &TuiExtensions,
    runtime: &mut Option<LashSession>,
    effects: Vec<TuiHostEffect>,
) {
    let mut local_effects = Vec::new();
    for effect in effects {
        match effect {
            TuiHostEffect::RunPluginCommand { name, args } => {
                let Some(session) = runtime.as_ref() else {
                    push_system_message(app, "No active session for plugin command.".to_string());
                    continue;
                };
                match session
                    .plugin_operations()
                    .run_command_raw(&name, args)
                    .await
                {
                    Ok(receipt) => replay_plugin_operation_receipt(
                        app,
                        ui_extensions,
                        &mut local_effects,
                        receipt.output,
                        receipt.events,
                    ),
                    Err(err) => push_system_message(app, format!("Plugin command failed: {err}")),
                }
            }
            TuiHostEffect::RunPluginTask { name, args } => {
                let Some(session) = runtime.as_ref() else {
                    push_system_message(app, "No active session for plugin task.".to_string());
                    continue;
                };
                match session.plugin_operations().run_task_raw(&name, args).await {
                    Ok(receipt) => replay_plugin_operation_receipt(
                        app,
                        ui_extensions,
                        &mut local_effects,
                        receipt.output,
                        receipt.events,
                    ),
                    Err(err) => push_system_message(app, format!("Plugin task failed: {err}")),
                }
            }
            TuiHostEffect::QueueTurn { input } => {
                if let Some(session) = runtime.as_ref() {
                    if let Err(err) = session.enqueue(TurnInput::text(input)).send().await {
                        push_system_message(app, format!("Failed to queue turn: {err}"));
                    }
                } else {
                    push_system_message(app, "No active session for queued turn.".to_string());
                }
            }
            TuiHostEffect::QueuePreparedTurn { effective_text, .. } => {
                if let Some(session) = runtime.as_ref() {
                    if let Err(err) = session
                        .enqueue(TurnInput::text(effective_text))
                        .send()
                        .await
                    {
                        push_system_message(app, format!("Failed to queue turn: {err}"));
                    }
                } else {
                    push_system_message(app, "No active session for queued turn.".to_string());
                }
            }
            other => local_effects.push(other),
        }
    }
    apply_ui_host_effects(app, local_effects);
}

fn replay_plugin_operation_receipt(
    app: &mut App,
    ui_extensions: &TuiExtensions,
    local_effects: &mut Vec<TuiHostEffect>,
    output: serde_json::Value,
    events: Vec<PluginOwned<PluginRuntimeEvent>>,
) {
    push_plugin_operation_message(app, &output);
    for owned in events {
        local_effects.extend(
            ui_extensions.effects_for_turn_event(&TurnEvent::PluginRuntime {
                plugin_id: owned.plugin_id,
                event: owned.value,
            }),
        );
    }
}

fn push_plugin_operation_message(app: &mut App, output: &serde_json::Value) {
    if let Some(message) = output.get("message").and_then(|value| value.as_str()) {
        push_system_message(app, message.to_string());
    }
}

pub(crate) async fn collect_ui_snapshot(
    session: lash::LashSession,
) -> crate::event::UiSnapshotResult {
    let started = std::time::Instant::now();
    let mut diagnostics = Vec::new();
    let processes = match session.processes().list().await {
        Ok(tasks) => Some(tasks),
        Err(err) => {
            diagnostics.push(format!("process snapshot failed: {err}"));
            None
        }
    };
    crate::event::UiSnapshotResult {
        effects: Vec::new(),
        processes,
        duration: started.elapsed(),
        timed_out: false,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desktop_notification_effect_respects_focus() {
        let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
        app.focused = true;

        apply_ui_host_effects(
            &mut app,
            vec![TuiHostEffect::DesktopNotification {
                title: "lash".into(),
                body: "Response complete".into(),
                only_when_unfocused: true,
            }],
        );
    }

    #[test]
    fn plugin_operation_effect_without_runtime_reports_requirement() {
        let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());

        apply_ui_host_effects(
            &mut app,
            vec![TuiHostEffect::RunPluginCommand {
                name: "plugin.command".into(),
                args: serde_json::json!({}),
            }],
        );

        assert!(matches!(
            app.timeline.last(),
            Some(UiTimelineItem::SystemMessage(message))
                if message == "Plugin operation requests require a live runtime session."
        ));
    }

    #[tokio::test]
    async fn plugin_operation_effect_with_missing_runtime_reports_no_active_session() {
        let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
        let ui_extensions = TuiExtensions::default();
        let mut runtime = None;

        apply_ui_host_effects_with_runtime(
            &mut app,
            &ui_extensions,
            &mut runtime,
            vec![TuiHostEffect::RunPluginTask {
                name: "plugin.task".into(),
                args: serde_json::json!({}),
            }],
        )
        .await;

        assert!(matches!(
            app.timeline.last(),
            Some(UiTimelineItem::SystemMessage(message))
                if message == "No active session for plugin task."
        ));
    }
}
