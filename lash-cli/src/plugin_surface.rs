use std::collections::BTreeMap;

use lash::PluginSurfaceEvent;

use crate::app::{
    PlanDockItem, PlanDockItemStatus, PlanDockState, PluginPanelBlock, UiTimeline, UiTimelineItem,
};

pub struct PluginSurfaceMutation {
    pub blocks_changed: bool,
    pub indicators_changed: bool,
    /// Side-channel update for the bottom-anchored plan dock. When set,
    /// replaces the dock state wholesale (`Some(None)` clears the
    /// current dock, `Some(Some(state))` installs a fresh plan).
    pub plan_dock_change: Option<Option<PlanDockState>>,
}

impl PluginSurfaceMutation {
    fn blocks_changed(changed: bool) -> Self {
        Self {
            blocks_changed: changed,
            indicators_changed: false,
            plan_dock_change: None,
        }
    }

    fn indicators_changed(changed: bool) -> Self {
        Self {
            blocks_changed: false,
            indicators_changed: changed,
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

/// True when a `PanelUpsert` event targets the bottom plan dock rather
/// than the inline transcript. Only `update_plan` owns the dock — the
/// legacy `plan_mode` plugin's `panel` key is a file-path breadcrumb
/// and is synced from a `tick` hook that emits a `ClearPanel` whenever
/// plan mode is disabled, which would otherwise wipe the dock every
/// 250ms.
fn is_plan_dock_panel(plugin_id: &str) -> bool {
    plugin_id == "update_plan"
}

/// Parse a plan panel's body into checklist items. Accepts the following
/// markdown-ish line shapes (leading whitespace + optional `- `):
///
/// * `- [x]`, `- [X]`, `- [✓]` → [`PlanDockItemStatus::Done`]
/// * `- [~]`, `- [*]`, `- [-]`, `- [→]` → [`PlanDockItemStatus::Active`]
/// * `- [ ]` (or no checkbox) → [`PlanDockItemStatus::Pending`]
///
/// At most one item is promoted to `Active`. If the panel marked more
/// than one, the first one wins; the rest fall back to `Pending` so the
/// dock's "exactly one active row" invariant holds.
fn parse_plan_items(content: &str) -> Vec<PlanDockItem> {
    let mut out = Vec::new();
    let mut active_seen = false;
    for raw in content.lines() {
        let line = raw.trim_start();
        let line = line
            .strip_prefix("- ")
            .or_else(|| line.strip_prefix("* "))
            .unwrap_or(line);
        if line.is_empty() {
            continue;
        }
        let (status, rest) = if let Some(rest) = line
            .strip_prefix("[x]")
            .or_else(|| line.strip_prefix("[X]"))
            .or_else(|| line.strip_prefix("[✓]"))
        {
            (PlanDockItemStatus::Done, rest)
        } else if let Some(rest) = line
            .strip_prefix("[~]")
            .or_else(|| line.strip_prefix("[*]"))
            .or_else(|| line.strip_prefix("[-]"))
            .or_else(|| line.strip_prefix("[→]"))
        {
            if active_seen {
                (PlanDockItemStatus::Pending, rest)
            } else {
                active_seen = true;
                (PlanDockItemStatus::Active, rest)
            }
        } else if let Some(rest) = line.strip_prefix("[ ]") {
            (PlanDockItemStatus::Pending, rest)
        } else {
            (PlanDockItemStatus::Pending, line)
        };
        let text = rest.trim().to_string();
        if !text.is_empty() {
            out.push(PlanDockItem { text, status });
        }
    }
    out
}

fn plan_state_from_panel(title: &str, content: &str) -> PlanDockState {
    PlanDockState {
        title: title.trim().to_string(),
        meta: None,
        items: parse_plan_items(content),
    }
}

pub fn apply_surface_event(
    blocks: &mut UiTimeline,
    indicators: &mut BTreeMap<String, String>,
    plan_dock: &Option<PlanDockState>,
    plugin_id: &str,
    event: PluginSurfaceEvent,
) -> PluginSurfaceMutation {
    match event {
        PluginSurfaceEvent::ModeIndicatorUpsert { key, label } => {
            let next_key = surface_key(plugin_id, &key);
            let changed = indicators.get(&next_key) != Some(&label);
            indicators.insert(next_key, label);
            PluginSurfaceMutation::indicators_changed(changed)
        }
        PluginSurfaceEvent::ModeIndicatorClear { key } => {
            let changed = indicators.remove(&surface_key(plugin_id, &key)).is_some();
            PluginSurfaceMutation::indicators_changed(changed)
        }
        PluginSurfaceEvent::PanelUpsert {
            key,
            title,
            content,
        } => {
            if is_plan_dock_panel(plugin_id) {
                let next = plan_state_from_panel(&title, &content);
                // Auto-hide: once every item is done (or the plan was
                // cleared to empty), drop the dock instead of rendering
                // a full column of ticked boxes.
                let all_done = !next.items.is_empty()
                    && next
                        .items
                        .iter()
                        .all(|item| item.status == PlanDockItemStatus::Done);
                if next.items.is_empty() || all_done {
                    tracing::info!(
                        target: "lash::plan_dock",
                        plugin_id = plugin_id,
                        items = next.items.len(),
                        all_done = all_done,
                        "plan panel upsert cleared dock (all items done)",
                    );
                    return PluginSurfaceMutation::plan_dock_update(None);
                }
                tracing::info!(
                    target: "lash::plan_dock",
                    plugin_id = plugin_id,
                    items = next.items.len(),
                    "routing panel upsert to plan dock",
                );
                return PluginSurfaceMutation::plan_dock_update(Some(next));
            }

            let target_key = surface_key(plugin_id, &key);
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
                PluginSurfaceMutation::blocks_changed(changed)
            } else {
                blocks.push(UiTimelineItem::PluginPanel(PluginPanelBlock {
                    plugin_id: plugin_id.to_string(),
                    key,
                    title,
                    content,
                }));
                PluginSurfaceMutation::blocks_changed(true)
            }
        }
        PluginSurfaceEvent::PanelAppend { key, content } => {
            // If a plan dock is active and owned by this plugin, allow
            // append to extend the checklist instead of the inline
            // panel. Otherwise fall through to the normal panel path.
            if is_plan_dock_panel(plugin_id)
                && let Some(existing) = plan_dock.clone()
            {
                let mut next = existing;
                let new_items = parse_plan_items(&content);
                if new_items.is_empty() {
                    return PluginSurfaceMutation::plan_dock_update(None);
                }
                next.items.extend(new_items);
                return PluginSurfaceMutation::plan_dock_update(Some(next));
            }

            let target_key = surface_key(plugin_id, &key);
            if let Some(existing) = blocks.iter_mut().find_map(|block| match block {
                UiTimelineItem::PluginPanel(panel)
                    if surface_key(&panel.plugin_id, &panel.key) == target_key =>
                {
                    Some(panel)
                }
                _ => None,
            }) {
                if content.is_empty() {
                    PluginSurfaceMutation::blocks_changed(false)
                } else {
                    existing.content.push_str(&content);
                    PluginSurfaceMutation::blocks_changed(true)
                }
            } else {
                PluginSurfaceMutation::blocks_changed(false)
            }
        }
        PluginSurfaceEvent::PanelClear { key } => {
            if is_plan_dock_panel(plugin_id) && plan_dock.is_some() {
                return PluginSurfaceMutation::plan_dock_update(None);
            }

            let original_len = blocks.len();
            let target_key = surface_key(plugin_id, &key);
            blocks.retain(|block| match block {
                UiTimelineItem::PluginPanel(panel) => {
                    surface_key(&panel.plugin_id, &panel.key) != target_key
                }
                _ => true,
            });
            PluginSurfaceMutation::blocks_changed(blocks.len() != original_len)
        }
        PluginSurfaceEvent::Status { .. } => PluginSurfaceMutation::blocks_changed(false),
        PluginSurfaceEvent::Custom { .. } => PluginSurfaceMutation::blocks_changed(false),
    }
}

pub fn desktop_notification_effect(event: &PluginSurfaceEvent) -> Option<lash_ui::UiHostEffect> {
    let PluginSurfaceEvent::Custom { name, payload } = event else {
        return None;
    };
    if name != "desktop_notification" {
        return None;
    }
    Some(lash_ui::UiHostEffect::DesktopNotification {
        title: payload
            .get("title")
            .and_then(|value| value.as_str())
            .unwrap_or("lash")
            .to_string(),
        body: payload
            .get("body")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string(),
        only_when_unfocused: payload
            .get("only_when_unfocused")
            .and_then(|value| value.as_bool())
            .unwrap_or(true),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_plan_items_recognises_checkbox_variants() {
        let content =
            "- [x] Fix ink 4 contrast\n- [~] Rework anatomy\n- [ ] Audit queries\n- Rename graph\n";
        let items = parse_plan_items(content);
        assert_eq!(items.len(), 4);
        assert_eq!(items[0].status, PlanDockItemStatus::Done);
        assert_eq!(items[0].text, "Fix ink 4 contrast");
        assert_eq!(items[1].status, PlanDockItemStatus::Active);
        assert_eq!(items[1].text, "Rework anatomy");
        assert_eq!(items[2].status, PlanDockItemStatus::Pending);
        assert_eq!(items[3].status, PlanDockItemStatus::Pending);
        assert_eq!(items[3].text, "Rename graph");
    }

    #[test]
    fn parse_plan_items_demotes_extra_active_rows() {
        let content = "- [~] first\n- [*] second\n- [-] third";
        let items = parse_plan_items(content);
        let actives = items
            .iter()
            .filter(|item| item.status == PlanDockItemStatus::Active)
            .count();
        assert_eq!(actives, 1, "only the first active marker wins");
    }

    #[test]
    fn panel_upsert_from_update_plan_routes_to_dock() {
        let mut blocks = UiTimeline::default();
        let mut indicators = BTreeMap::new();
        let dock: Option<PlanDockState> = None;

        let mutation = apply_surface_event(
            &mut blocks,
            &mut indicators,
            &dock,
            "update_plan",
            PluginSurfaceEvent::PanelUpsert {
                key: "plan".into(),
                title: "PLAN".into(),
                content: "- [x] done\n- [~] active\n- [ ] later".into(),
            },
        );

        assert!(blocks.is_empty(), "plan panel should not land in history");
        let next = mutation
            .plan_dock_change
            .expect("plan_dock_change should be set")
            .expect("should carry a state");
        assert_eq!(next.title, "PLAN");
        assert_eq!(next.items.len(), 3);
        assert_eq!(next.items[1].status, PlanDockItemStatus::Active);
    }

    #[test]
    fn panel_clear_from_update_plan_clears_dock() {
        let mut blocks = UiTimeline::default();
        let mut indicators = BTreeMap::new();
        let dock = Some(PlanDockState {
            title: "PLAN".into(),
            meta: None,
            items: vec![PlanDockItem {
                text: "x".into(),
                status: PlanDockItemStatus::Pending,
            }],
        });

        let mutation = apply_surface_event(
            &mut blocks,
            &mut indicators,
            &dock,
            "update_plan",
            PluginSurfaceEvent::PanelClear { key: "plan".into() },
        );

        assert_eq!(mutation.plan_dock_change, Some(None));
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
            PluginSurfaceEvent::PanelUpsert {
                key: "plan".into(),
                title: "PLAN".into(),
                content: "- [x] chop\n- [x] simmer\n- [x] plate".into(),
            },
        );

        assert_eq!(
            mutation.plan_dock_change,
            Some(None),
            "dock should auto-hide once every item is completed"
        );
    }

    /// Regression: the `plan_mode` UI extension's tick hook emits
    /// `ClearPanel { plugin_id: "plan_mode", key: "panel" }` every 250 ms
    /// whenever plan mode is disabled. That clear must NOT touch the
    /// sticky dock owned by `update_plan`, or the dock flickers away a
    /// quarter-second after every tool call.
    #[test]
    fn plan_mode_clear_does_not_wipe_update_plan_dock() {
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
            PluginSurfaceEvent::PanelClear {
                key: "panel".into(),
            },
        );

        assert!(
            mutation.plan_dock_change.is_none(),
            "plan_mode clears must not touch the update_plan dock"
        );
    }

    #[test]
    fn unrelated_panel_still_becomes_inline_block() {
        let mut blocks = UiTimeline::default();
        let mut indicators = BTreeMap::new();
        let dock: Option<PlanDockState> = None;

        let mutation = apply_surface_event(
            &mut blocks,
            &mut indicators,
            &dock,
            "some_plugin",
            PluginSurfaceEvent::PanelUpsert {
                key: "x".into(),
                title: "Other".into(),
                content: "hello".into(),
            },
        );

        assert!(mutation.blocks_changed);
        assert!(mutation.plan_dock_change.is_none());
        assert_eq!(blocks.len(), 1);
    }
}
