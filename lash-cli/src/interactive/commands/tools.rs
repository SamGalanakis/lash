use std::sync::Arc;

use lash::*;

use crate::app::App;
use crate::{hash12, push_system_message};

use super::super::runtime::{apply_pending_reconfigure, parse_kv_args, register_builtin_tool};

pub(super) fn handle_tools(
    raw: Option<String>,
    app: &mut App,
    dynamic_tools: &Arc<DynamicToolProvider>,
    desired_dynamic: &mut DynamicStateSnapshot,
    pending_reconfigure: &mut bool,
    current_execution_mode: ExecutionMode,
) -> anyhow::Result<bool> {
    let raw = raw.unwrap_or_default();
    let raw_trim = raw.trim();
    if raw_trim.is_empty() {
        let active = dynamic_tools.export_state();
        let mut lines = vec![
            format!("Dynamic tools (generation {}):", active.base_generation),
            format!(
                "Pending reconfigure: {}",
                if *pending_reconfigure { "yes" } else { "no" }
            ),
        ];
        for (name, spec) in &desired_dynamic.tools {
            let enabled = desired_dynamic.enabled_tools.contains(name);
            lines.push(format!(
                "  - {} [{}] adapter={}{}",
                name,
                spec.definition.returns,
                spec.adapter_id,
                if enabled { " (enabled)" } else { " (disabled)" }
            ));
        }
        if desired_dynamic.tools.is_empty() {
            lines.push("  (none)".to_string());
        }
        push_system_message(app, lines.join("\n"));
        return Ok(false);
    }

    let mut parts = raw_trim.split_whitespace();
    let sub = parts.next().unwrap_or_default();
    match sub {
        "add" => {
            let mut add_parts = raw_trim.splitn(4, ' ');
            let _ = add_parts.next();
            let Some(name) = add_parts.next() else {
                push_system_message(app, "Usage: /tools add <name> <handler> [description]");
                return Ok(false);
            };
            let Some(handler_id) = add_parts.next() else {
                push_system_message(app, "Usage: /tools add <name> <handler> [description]");
                return Ok(false);
            };
            let description = add_parts.next().map(|v| v.trim().to_string());
            match register_builtin_tool(
                dynamic_tools,
                name,
                handler_id,
                description,
                current_execution_mode,
            ) {
                Ok(def) => {
                    desired_dynamic.tools.insert(
                        name.to_string(),
                        DynamicToolSpec {
                            definition: def,
                            adapter_id: "inprocess".to_string(),
                        },
                    );
                    desired_dynamic.enabled_tools.insert(name.to_string());
                    *pending_reconfigure = true;
                    push_system_message(
                        app,
                        format!(
                            "Tool `{}` staged with handler `{}`. Apply with `/reconfigure apply` or send the next turn.",
                            name, handler_id
                        ),
                    );
                }
                Err(e) => push_system_message(app, e),
            }
        }
        "rm" | "remove" => {
            let Some(name) = parts.next() else {
                push_system_message(app, "Usage: /tools rm <name>");
                return Ok(false);
            };
            if desired_dynamic.tools.remove(name).is_some() {
                desired_dynamic.enabled_tools.remove(name);
                *pending_reconfigure = true;
                push_system_message(app, format!("Tool `{name}` staged for removal."));
            } else {
                push_system_message(app, format!("Tool `{name}` not found."));
            }
        }
        "update" => {
            let mut update_parts = raw_trim.splitn(3, ' ');
            let _ = update_parts.next();
            let Some(name) = update_parts.next() else {
                push_system_message(app, "Usage: /tools update <name> key=value ...");
                return Ok(false);
            };
            let kv_raw = update_parts.next().unwrap_or_default();
            let kv = parse_kv_args(kv_raw);
            let Some(spec) = desired_dynamic.tools.get_mut(name) else {
                push_system_message(app, format!("Tool `{name}` not found."));
                return Ok(false);
            };
            if let Some(desc) = kv.get("description") {
                spec.definition.description = desc.clone();
            }
            if let Some(returns) = kv.get("returns") {
                spec.definition.returns = returns.clone();
            }
            if let Some(inject) = kv.get("injected") {
                spec.definition.injected = inject == "true";
            }
            *pending_reconfigure = true;
            push_system_message(app, format!("Tool `{name}` staged for update."));
        }
        "enable" => {
            let Some(name) = parts.next() else {
                push_system_message(app, "Usage: /tools enable <name>");
                return Ok(false);
            };
            if desired_dynamic.tools.contains_key(name) {
                desired_dynamic.enabled_tools.insert(name.to_string());
                *pending_reconfigure = true;
                push_system_message(app, format!("Tool `{name}` staged for enable."));
            } else {
                push_system_message(app, format!("Tool `{name}` not found."));
            }
        }
        "disable" => {
            let Some(name) = parts.next() else {
                push_system_message(app, "Usage: /tools disable <name>");
                return Ok(false);
            };
            if desired_dynamic.tools.contains_key(name) {
                desired_dynamic.enabled_tools.remove(name);
                *pending_reconfigure = true;
                push_system_message(app, format!("Tool `{name}` staged for disable."));
            } else {
                push_system_message(app, format!("Tool `{name}` not found."));
            }
        }
        _ => push_system_message(
            app,
            "Unknown /tools subcommand. Try: add, rm, update, enable, disable",
        ),
    }
    Ok(false)
}

pub(super) async fn handle_reconfigure(
    raw: Option<String>,
    app: &mut App,
    dynamic_tools: &Arc<DynamicToolProvider>,
    runtime: &mut Option<LashRuntime>,
    desired_dynamic: &mut DynamicStateSnapshot,
    pending_reconfigure: &mut bool,
    toolset_hash: &mut String,
) -> anyhow::Result<bool> {
    let action = raw.unwrap_or_else(|| "status".to_string());
    match action.trim() {
        "" | "status" => {
            push_system_message(
                app,
                format!(
                    "Reconfigure status: pending={} current_generation={} base_generation={}",
                    pending_reconfigure,
                    dynamic_tools.generation(),
                    desired_dynamic.base_generation
                ),
            );
        }
        "clear" => {
            *desired_dynamic = dynamic_tools.export_state();
            *pending_reconfigure = false;
            push_system_message(app, "Cleared pending dynamic runtime changes.");
        }
        "apply" => match apply_pending_reconfigure(
            dynamic_tools,
            desired_dynamic,
            pending_reconfigure,
            runtime,
        )
        .await
        {
            Ok(generation) => {
                *toolset_hash = hash12(
                    &serde_json::to_vec(&dynamic_tools.definitions())
                        .unwrap_or_else(|_| b"[]".to_vec()),
                );
                push_system_message(
                    app,
                    format!(
                        "Dynamic runtime reconfigured successfully (generation {}).",
                        generation
                    ),
                )
            }
            Err(e) => push_system_message(app, format!("Reconfigure failed: {e}")),
        },
        _ => push_system_message(
            app,
            "Unknown /reconfigure action. Try: status, apply, clear",
        ),
    }
    Ok(false)
}
