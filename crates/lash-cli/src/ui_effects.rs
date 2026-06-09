//! Applying plugin/UI-extension host effects to the TUI [`App`], plus the
//! shared system-message and UI-snapshot helpers.

use lash_tui_extensions::{TuiExtensionContext, TuiExtensions, TuiHostEffect};

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

pub(crate) async fn collect_ui_snapshot(
    session: lash::LashSession,
    ui_extensions: &TuiExtensions,
) -> crate::event::UiSnapshotResult {
    let started = std::time::Instant::now();
    let mut diagnostics = Vec::new();
    let processes = match session.process_control().list().await {
        Ok(tasks) => Some(tasks),
        Err(err) => {
            diagnostics.push(format!("process snapshot failed: {err}"));
            None
        }
    };
    let effects = match ui_extensions
        .snapshot_all(TuiExtensionContext {
            actions: &session.plugin_actions(),
        })
        .await
    {
        Ok(effects) => effects,
        Err(err) => {
            diagnostics.push(format!("Failed to snapshot UI extensions: {err}"));
            Vec::new()
        }
    };
    crate::event::UiSnapshotResult {
        effects,
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
}
