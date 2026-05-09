use lash::*;
use lash_embed::LashSession;

use crate::app::App;
use crate::{hash12, push_system_message};

use super::super::runtime::{apply_pending_reconfigure, parse_kv_args};

fn availability_label(availability: ToolAvailability) -> &'static str {
    match availability {
        ToolAvailability::Hidden => "hidden",
        ToolAvailability::Discoverable => "discoverable",
        ToolAvailability::Callable => "callable",
        ToolAvailability::Documented => "documented",
    }
}

fn parse_availability(raw: &str) -> Option<ToolAvailability> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "hidden" => Some(ToolAvailability::Hidden),
        "discoverable" => Some(ToolAvailability::Discoverable),
        "callable" => Some(ToolAvailability::Callable),
        "documented" => Some(ToolAvailability::Documented),
        _ => None,
    }
}

fn update_documentation_state(availability: ToolAvailability, injected: bool) -> ToolAvailability {
    match (availability, injected) {
        (ToolAvailability::Documented, false) => ToolAvailability::Callable,
        (ToolAvailability::Callable, true) => ToolAvailability::Documented,
        (other, _) => other,
    }
}

fn output_schema_label(schema: &serde_json::Value) -> String {
    match schema.get("type").and_then(serde_json::Value::as_str) {
        Some("string") => "str".to_string(),
        Some("integer") => "int".to_string(),
        Some("number") => "float".to_string(),
        Some("boolean") => "bool".to_string(),
        Some("object") => "record".to_string(),
        Some("array") => "list".to_string(),
        Some("null") => "null".to_string(),
        _ => "any".to_string(),
    }
}

fn output_schema_from_hint(hint: &str) -> serde_json::Value {
    match hint.trim() {
        "str" | "string" => serde_json::json!({ "type": "string" }),
        "int" | "integer" => serde_json::json!({ "type": "integer" }),
        "float" | "number" => serde_json::json!({ "type": "number" }),
        "bool" | "boolean" => serde_json::json!({ "type": "boolean" }),
        "dict" | "record" | "json" => {
            serde_json::json!({ "type": "object", "additionalProperties": true })
        }
        "list" | "array" => serde_json::json!({ "type": "array", "items": {} }),
        "null" | "None" => serde_json::json!({ "type": "null" }),
        _ => serde_json::json!({}),
    }
}

pub(super) async fn handle_tools(
    raw: Option<String>,
    app: &mut App,
    _runtime: &Option<lash_embed::LashSession>,
    desired_dynamic: &mut DynamicStateSnapshot,
    pending_reconfigure: &mut bool,
    current_execution_mode: ExecutionMode,
) -> anyhow::Result<bool> {
    let raw = raw.unwrap_or_default();
    let raw_trim = raw.trim();
    if raw_trim.is_empty() {
        let mut lines = vec![
            format!(
                "Dynamic tools (generation {}):",
                desired_dynamic.base_generation
            ),
            format!(
                "Pending reconfigure: {}",
                if *pending_reconfigure { "yes" } else { "no" }
            ),
        ];
        for (name, spec) in &desired_dynamic.tools {
            let availability = spec
                .definition
                .effective_availability(&current_execution_mode);
            lines.push(format!(
                "  - {} [{}] adapter={} availability={}",
                name,
                output_schema_label(&spec.definition.output_schema),
                spec.adapter_id,
                availability_label(availability)
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
        "rm" | "remove" => {
            let Some(name) = parts.next() else {
                push_system_message(app, "Usage: /tools rm <name>");
                return Ok(false);
            };
            if desired_dynamic.tools.remove(name).is_some() {
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
                spec.definition.output_schema = output_schema_from_hint(returns);
            }
            if let Some(raw) = kv.get("availability") {
                let Some(availability) = parse_availability(raw) else {
                    push_system_message(
                        app,
                        "Invalid availability. Use: hidden, discoverable, callable, documented",
                    );
                    return Ok(false);
                };
                spec.definition.availability_override = Some(availability);
            }
            if let Some(inject) = kv.get("injected") {
                let injected = inject == "true";
                let availability = spec
                    .definition
                    .effective_availability(&current_execution_mode);
                spec.definition.availability_override =
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
            if desired_dynamic.tools.contains_key(name) {
                desired_dynamic
                    .tools
                    .get_mut(name)
                    .expect("checked contains_key")
                    .definition
                    .availability_override = Some(ToolAvailability::Callable);
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
                desired_dynamic
                    .tools
                    .get_mut(name)
                    .expect("checked contains_key")
                    .definition
                    .availability_override = Some(ToolAvailability::Hidden);
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
                    desired_dynamic.base_generation,
                    desired_dynamic.base_generation
                ),
            );
        }
        "clear" => {
            if let Some(session) = runtime.as_ref() {
                *desired_dynamic = session.tool_state().await?;
            }
            *pending_reconfigure = false;
            push_system_message(app, "Cleared pending dynamic runtime changes.");
        }
        "apply" => match apply_pending_reconfigure(desired_dynamic, pending_reconfigure, runtime)
            .await
        {
            Ok(generation) => {
                let definitions = match runtime.as_ref() {
                    Some(session) => session.active_tool_definitions().await.unwrap_or_default(),
                    None => desired_dynamic
                        .tools
                        .values()
                        .map(|spec| spec.definition.clone())
                        .collect(),
                };
                *toolset_hash =
                    hash12(&serde_json::to_vec(&definitions).unwrap_or_else(|_| b"[]".to_vec()));
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
