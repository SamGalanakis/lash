use std::collections::BTreeMap;

use lash_core::PluginSurfaceEvent;

use crate::app::{DisplayBlock, PluginPanelBlock};

pub struct PluginSurfaceMutation {
    pub blocks_changed: bool,
    pub indicators_changed: bool,
}

pub fn surface_key(plugin_id: &str, key: &str) -> String {
    format!("{plugin_id}:{key}")
}

pub fn apply_surface_event(
    blocks: &mut Vec<DisplayBlock>,
    indicators: &mut BTreeMap<String, String>,
    plugin_id: &str,
    event: PluginSurfaceEvent,
) -> PluginSurfaceMutation {
    match event {
        PluginSurfaceEvent::ModeIndicatorUpsert { key, label } => {
            let next_key = surface_key(plugin_id, &key);
            let changed = indicators.get(&next_key) != Some(&label);
            indicators.insert(next_key, label);
            PluginSurfaceMutation {
                blocks_changed: false,
                indicators_changed: changed,
            }
        }
        PluginSurfaceEvent::ModeIndicatorClear { key } => PluginSurfaceMutation {
            blocks_changed: false,
            indicators_changed: indicators.remove(&surface_key(plugin_id, &key)).is_some(),
        },
        PluginSurfaceEvent::PanelUpsert {
            key,
            title,
            content,
        } => {
            let target_key = surface_key(plugin_id, &key);
            if let Some(existing) = blocks.iter_mut().find_map(|block| match block {
                DisplayBlock::PluginPanel(panel)
                    if surface_key(&panel.plugin_id, &panel.key) == target_key =>
                {
                    Some(panel)
                }
                _ => None,
            }) {
                let changed = existing.title != title || existing.content != content;
                existing.title = title;
                existing.content = content;
                PluginSurfaceMutation {
                    blocks_changed: changed,
                    indicators_changed: false,
                }
            } else {
                blocks.push(DisplayBlock::PluginPanel(PluginPanelBlock {
                    plugin_id: plugin_id.to_string(),
                    key,
                    title,
                    content,
                }));
                PluginSurfaceMutation {
                    blocks_changed: true,
                    indicators_changed: false,
                }
            }
        }
        PluginSurfaceEvent::PanelAppend { key, content } => {
            let target_key = surface_key(plugin_id, &key);
            if let Some(existing) = blocks.iter_mut().find_map(|block| match block {
                DisplayBlock::PluginPanel(panel)
                    if surface_key(&panel.plugin_id, &panel.key) == target_key =>
                {
                    Some(panel)
                }
                _ => None,
            }) {
                if content.is_empty() {
                    PluginSurfaceMutation {
                        blocks_changed: false,
                        indicators_changed: false,
                    }
                } else {
                    existing.content.push_str(&content);
                    PluginSurfaceMutation {
                        blocks_changed: true,
                        indicators_changed: false,
                    }
                }
            } else {
                PluginSurfaceMutation {
                    blocks_changed: false,
                    indicators_changed: false,
                }
            }
        }
        PluginSurfaceEvent::PanelClear { key } => {
            let original_len = blocks.len();
            let target_key = surface_key(plugin_id, &key);
            blocks.retain(|block| match block {
                DisplayBlock::PluginPanel(panel) => {
                    surface_key(&panel.plugin_id, &panel.key) != target_key
                }
                _ => true,
            });
            PluginSurfaceMutation {
                blocks_changed: blocks.len() != original_len,
                indicators_changed: false,
            }
        }
        PluginSurfaceEvent::Custom { .. } => PluginSurfaceMutation {
            blocks_changed: false,
            indicators_changed: false,
        },
    }
}
