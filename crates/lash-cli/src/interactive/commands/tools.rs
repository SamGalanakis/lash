use lash::{LashSession, advanced::ExecutionMode, tools::ToolAvailability};
use lash_core::ToolState;

use crate::app::App;
use crate::{hash12, push_system_message};

use super::super::runtime::{apply_pending_reconfigure, parse_kv_args};

fn availability_label(availability: ToolAvailability) -> &'static str {
    match availability {
        ToolAvailability::Off => "off",
        ToolAvailability::Searchable => "searchable",
        ToolAvailability::Callable => "callable",
        ToolAvailability::Showcased => "showcased",
    }
}

fn parse_availability(raw: &str) -> Option<ToolAvailability> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "off" => Some(ToolAvailability::Off),
        "searchable" => Some(ToolAvailability::Searchable),
        "callable" => Some(ToolAvailability::Callable),
        "showcased" => Some(ToolAvailability::Showcased),
        _ => None,
    }
}

fn update_documentation_state(availability: ToolAvailability, injected: bool) -> ToolAvailability {
    match (availability, injected) {
        (ToolAvailability::Showcased, false) => ToolAvailability::Callable,
        (ToolAvailability::Callable, true) => ToolAvailability::Showcased,
        (other, _) => other,
    }
}

pub(super) async fn handle_tools(
    raw: Option<String>,
    app: &mut App,
    _runtime: &Option<lash::LashSession>,
    desired_tool_state: &mut ToolState,
    pending_reconfigure: &mut bool,
    _current_execution_mode: ExecutionMode,
) -> anyhow::Result<bool> {
    let raw = raw.unwrap_or_default();
    let raw_trim = raw.trim();
    if raw_trim.is_empty() {
        let mut lines = vec![
            format!("Tools (generation {}):", desired_tool_state.generation()),
            format!(
                "Pending reconfigure: {}",
                if *pending_reconfigure { "yes" } else { "no" }
            ),
        ];
        for (name, spec) in desired_tool_state.iter() {
            let availability = spec.manifest().effective_availability();
            lines.push(format!(
                "  - {} availability={}",
                name,
                availability_label(availability)
            ));
        }
        if desired_tool_state.is_empty() {
            lines.push("  (none)".to_string());
        }
        push_system_message(app, lines.join("\n"));
        return Ok(false);
    }

    let mut parts = raw_trim.split_whitespace();
    let sub = parts.next().unwrap_or_default();
    match sub {
        "rm" | "remove" => {
            let Some(name) = parts.next() else {
                push_system_message(app, "Usage: /tools rm <name>");
                return Ok(false);
            };
            if desired_tool_state.remove(name).is_some() {
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
            let Some(manifest) = desired_tool_state.manifest_mut(name) else {
                push_system_message(app, format!("Tool `{name}` not found."));
                return Ok(false);
            };
            if let Some(desc) = kv.get("description") {
                manifest.description = desc.clone();
            }
            if let Some(raw) = kv.get("availability") {
                let Some(availability) = parse_availability(raw) else {
                    push_system_message(
                        app,
                        "Invalid availability. Use: off, searchable, callable, showcased",
                    );
                    return Ok(false);
                };
                manifest.availability_override = Some(availability);
            }
            if let Some(inject) = kv.get("injected") {
                let injected = inject == "true";
                let availability = manifest.effective_availability();
                manifest.availability_override =
                    Some(update_documentation_state(availability, injected));
            }
            *pending_reconfigure = true;
            push_system_message(app, format!("Tool `{name}` staged for update."));
        }
        "enable" => {
            let Some(name) = parts.next() else {
                push_system_message(app, "Usage: /tools enable <name>");
                return Ok(false);
            };
            if desired_tool_state.contains(name) {
                desired_tool_state
                    .set_availability(name, Some(ToolAvailability::Callable))
                    .expect("checked contains");
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
            if desired_tool_state.contains(name) {
                desired_tool_state
                    .set_availability(name, Some(ToolAvailability::Off))
                    .expect("checked contains");
                *pending_reconfigure = true;
                push_system_message(app, format!("Tool `{name}` staged for disable."));
            } else {
                push_system_message(app, format!("Tool `{name}` not found."));
            }
        }
        _ => push_system_message(
            app,
            "Unknown /tools subcommand. Try: rm, update, enable, disable",
        ),
    }
    Ok(false)
}

pub(super) async fn handle_reconfigure(
    raw: Option<String>,
    app: &mut App,
    runtime: &mut Option<LashSession>,
    desired_tool_state: &mut ToolState,
    pending_reconfigure: &mut bool,
    toolset_hash: &mut String,
) -> anyhow::Result<bool> {
    let action = raw.unwrap_or_else(|| "status".to_string());
    match action.trim() {
        "" | "status" => {
            push_system_message(
                app,
                format!(
                    "Reconfigure status: pending={} current_generation={} generation={}",
                    pending_reconfigure,
                    desired_tool_state.generation(),
                    desired_tool_state.generation()
                ),
            );
        }
        "clear" => {
            if let Some(session) = runtime.as_ref() {
                *desired_tool_state = session.control().tools().state().await?;
            }
            *pending_reconfigure = false;
            push_system_message(app, "Cleared pending tool registry changes.");
        }
        "apply" => {
            match apply_pending_reconfigure(desired_tool_state, pending_reconfigure, runtime).await
            {
                Ok(generation) => {
                    let definitions = match runtime.as_ref() {
                        Some(session) => session
                            .control()
                            .tools()
                            .active_definitions()
                            .await
                            .unwrap_or_default(),
                        None => desired_tool_state.tool_manifests(),
                    };
                    *toolset_hash = hash12(
                        &serde_json::to_vec(&definitions).unwrap_or_else(|_| b"[]".to_vec()),
                    );
                    push_system_message(
                        app,
                        format!(
                            "Tool registry reconfigured successfully (generation {}).",
                            generation
                        ),
                    )
                }
                Err(e) => push_system_message(app, format!("Reconfigure failed: {e}")),
            }
        }
        _ => push_system_message(
            app,
            "Unknown /reconfigure action. Try: status, apply, clear",
        ),
    }
    Ok(false)
}
