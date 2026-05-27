use std::collections::BTreeMap;

use lash_core::PluginRuntimeEvent;

use crate::app::{
    PlanDockItem, PlanDockItemStatus, PlanDockState, PluginPanelBlock, UiTimeline, UiTimelineItem,
};

pub struct PluginRuntimeMutation {
    pub blocks_changed: bool,
    pub indicators_changed: bool,
    /// Side-channel update for the bottom-anchored plan dock. When set,
    /// replaces the dock state wholesale (`Some(None)` clears the
    /// current dock, `Some(Some(state))` installs a fresh plan).
    pub plan_dock_change: Option<Option<PlanDockState>>,
}

impl PluginRuntimeMutation {
    fn blocks_changed(changed: bool) -> Self {
        Self {
            blocks_changed: changed,
            indicators_changed: false,
            plan_dock_change: None,
        }
    }

    fn plan_dock_update(update: Option<PlanDockState>) -> Self {
        Self {
            blocks_changed: false,
            indicators_changed: false,
            plan_dock_change: Some(update),
        }
    }
}

pub fn surface_key(plugin_id: &str, key: &str) -> String {
    format!("{plugin_id}:{key}")
}

pub(crate) fn event_renders_visible_output(event: &PluginRuntimeEvent) -> bool {
    match event {
        PluginRuntimeEvent::Custom { name, .. } => name == "plan_mode.state",
        PluginRuntimeEvent::Status { .. } => false,
    }
}

fn plan_state_from_snapshot(snapshot: lash_plugin_plan_mode::PlanSnapshot) -> PlanDockState {
    PlanDockState {
        title: "PLAN".to_string(),
        meta: snapshot.explanation,
        items: snapshot
            .plan
            .into_iter()
            .map(|item| PlanDockItem {
                text: item.step,
                status: match item.status.as_str() {
                    "completed" => PlanDockItemStatus::Done,
                    "in_progress" => PlanDockItemStatus::Active,
                    _ => PlanDockItemStatus::Pending,
                },
            })
            .collect(),
    }
}

fn upsert_panel(
    blocks: &mut UiTimeline,
    plugin_id: &str,
    key: &str,
    title: String,
    content: String,
) -> bool {
    let target_key = surface_key(plugin_id, key);
    if let Some(existing) = blocks.iter_mut().find_map(|block| match block {
        UiTimelineItem::PluginPanel(panel)
            if surface_key(&panel.plugin_id, &panel.key) == target_key =>
        {
            Some(panel)
        }
        _ => None,
    }) {
        let changed = existing.title != title || existing.content != content;
        existing.title = title;
        existing.content = content;
        changed
    } else {
        blocks.push(UiTimelineItem::PluginPanel(PluginPanelBlock {
            plugin_id: plugin_id.to_string(),
            key: key.to_string(),
            title,
            content,
        }));
        true
    }
}

fn clear_panel(blocks: &mut UiTimeline, plugin_id: &str, key: &str) -> bool {
    let original_len = blocks.len();
    let target_key = surface_key(plugin_id, key);
    blocks.retain(|block| match block {
        UiTimelineItem::PluginPanel(panel) => {
            surface_key(&panel.plugin_id, &panel.key) != target_key
        }
        _ => true,
    });
    blocks.len() != original_len
}

fn apply_plan_protocol_state(
    blocks: &mut UiTimeline,
    indicators: &mut BTreeMap<String, String>,
    payload: serde_json::Value,
) -> PluginRuntimeMutation {
    let Ok(status) =
        serde_json::from_value::<lash_plugin_plan_mode::PlanModeExternalStatus>(payload)
    else {
        return PluginRuntimeMutation::blocks_changed(false);
    };
    let indicator_key = surface_key("plan_mode", "mode");
    let indicators_changed = if status.enabled {
        let label = "plan".to_string();
        let changed = indicators.get(&indicator_key) != Some(&label);
        indicators.insert(indicator_key, label);
        changed
    } else {
        indicators.remove(&indicator_key).is_some()
    };
    let blocks_changed = if status.enabled {
        if let Some(path) = status.plan_path {
            upsert_panel(
                blocks,
                "plan_mode",
                "panel",
                "PLAN".to_string(),
                format!("Path: `{path}`"),
            )
        } else {
            clear_panel(blocks, "plan_mode", "panel")
        }
    } else {
        clear_panel(blocks, "plan_mode", "panel")
    };
    PluginRuntimeMutation {
        blocks_changed,
        indicators_changed,
        plan_dock_change: None,
    }
}

pub fn apply_surface_event(
    blocks: &mut UiTimeline,
    indicators: &mut BTreeMap<String, String>,
    _plan_dock: &Option<PlanDockState>,
    plugin_id: &str,
    event: PluginRuntimeEvent,
) -> PluginRuntimeMutation {
    match event {
        PluginRuntimeEvent::Status { .. } => PluginRuntimeMutation::blocks_changed(false),
        PluginRuntimeEvent::Custom { name, payload } => {
            if plugin_id == "update_plan" && name == "update_plan.snapshot" {
                let Ok(snapshot) =
                    serde_json::from_value::<lash_plugin_plan_mode::PlanSnapshot>(payload)
                else {
                    return PluginRuntimeMutation::blocks_changed(false);
                };
                let next = plan_state_from_snapshot(snapshot);
                let all_done = !next.items.is_empty()
                    && next
                        .items
                        .iter()
                        .all(|item| item.status == PlanDockItemStatus::Done);
                if next.items.is_empty() || all_done {
                    return PluginRuntimeMutation::plan_dock_update(None);
                }
                return PluginRuntimeMutation::plan_dock_update(Some(next));
            }
            if plugin_id == "plan_mode" && name == "plan_mode.state" {
                return apply_plan_protocol_state(blocks, indicators, payload);
            }
            PluginRuntimeMutation::blocks_changed(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_plugin_plan_mode::{PlanItem, PlanModeExternalStatus, PlanSnapshot};

    fn custom(name: &str, payload: impl serde::Serialize) -> PluginRuntimeEvent {
        PluginRuntimeEvent::Custom {
            name: name.to_string(),
            payload: serde_json::to_value(payload).expect("payload"),
        }
    }

    #[test]
    fn update_plan_snapshot_routes_to_dock() {
        let mut blocks = UiTimeline::default();
        let mut indicators = BTreeMap::new();
        let dock: Option<PlanDockState> = None;

        let mutation = apply_surface_event(
            &mut blocks,
            &mut indicators,
            &dock,
            "update_plan",
            custom(
                "update_plan.snapshot",
                PlanSnapshot {
                    explanation: None,
                    generation: 1,
                    plan: vec![
                        PlanItem {
                            step: "done".into(),
                            status: "completed".into(),
                        },
                        PlanItem {
                            step: "active".into(),
                            status: "in_progress".into(),
                        },
                        PlanItem {
                            step: "later".into(),
                            status: "pending".into(),
                        },
                    ],
                },
            ),
        );

        assert!(
            blocks.is_empty(),
            "plan snapshot should not land in history"
        );
        let next = mutation
            .plan_dock_change
            .expect("plan_dock_change should be set")
            .expect("should carry a state");
        assert_eq!(next.title, "PLAN");
        assert_eq!(next.items.len(), 3);
        assert_eq!(next.items[1].status, PlanDockItemStatus::Active);
    }

    #[test]
    fn all_completed_plan_auto_hides_dock() {
        let mut blocks = UiTimeline::default();
        let mut indicators = BTreeMap::new();
        let dock = Some(PlanDockState {
            title: "PLAN".into(),
            meta: None,
            items: vec![PlanDockItem {
                text: "in flight".into(),
                status: PlanDockItemStatus::Active,
            }],
        });

        let mutation = apply_surface_event(
            &mut blocks,
            &mut indicators,
            &dock,
            "update_plan",
            custom(
                "update_plan.snapshot",
                PlanSnapshot {
                    explanation: None,
                    generation: 2,
                    plan: vec![
                        PlanItem {
                            step: "chop".into(),
                            status: "completed".into(),
                        },
                        PlanItem {
                            step: "simmer".into(),
                            status: "completed".into(),
                        },
                        PlanItem {
                            step: "plate".into(),
                            status: "completed".into(),
                        },
                    ],
                },
            ),
        );

        assert_eq!(
            mutation.plan_dock_change,
            Some(None),
            "dock should auto-hide once every item is completed"
        );
    }

    #[test]
    fn plan_protocol_state_updates_indicator_and_panel_without_touching_dock() {
        let mut blocks = UiTimeline::default();
        let mut indicators = BTreeMap::new();
        let dock = Some(PlanDockState {
            title: "PLAN".into(),
            meta: None,
            items: vec![PlanDockItem {
                text: "Patch layout".into(),
                status: PlanDockItemStatus::Active,
            }],
        });

        let mutation = apply_surface_event(
            &mut blocks,
            &mut indicators,
            &dock,
            "plan_mode",
            custom(
                "plan_mode.state",
                PlanModeExternalStatus {
                    session_id: "root".into(),
                    enabled: true,
                    plan_path: Some(".lash/plans/root.md".into()),
                },
            ),
        );

        assert!(mutation.blocks_changed);
        assert!(mutation.indicators_changed);
        assert!(mutation.plan_dock_change.is_none());
        assert_eq!(indicators.get("plan_mode:mode"), Some(&"plan".to_string()));
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn unrelated_custom_event_is_ignored() {
        let mut blocks = UiTimeline::default();
        let mut indicators = BTreeMap::new();
        let dock: Option<PlanDockState> = None;

        let mutation = apply_surface_event(
            &mut blocks,
            &mut indicators,
            &dock,
            "some_plugin",
            custom("some.event", serde_json::json!({"value": 1})),
        );

        assert!(!mutation.blocks_changed);
        assert!(mutation.plan_dock_change.is_none());
        assert!(blocks.is_empty());
    }
}
