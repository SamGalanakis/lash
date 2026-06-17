//! Compiled sources for the Rust snippets on `docs/example-agent-workbench.html`.

use lash::plugins::{PluginError, PluginRegistrar};
use lash::triggers::TriggerEvent;

// docs:start:workbench-resources
fn field(name: &str, ty: lashlang::TypeExpr) -> lashlang::TypeField {
    lashlang::TypeField {
        name: name.into(),
        ty,
        optional: false,
    }
}

fn schedule_config_type() -> lashlang::TypeExpr {
    lashlang::TypeExpr::Object(vec![
        field("expr", lashlang::TypeExpr::Str),
        lashlang::TypeField {
            name: "tz".into(),
            ty: lashlang::TypeExpr::Str,
            optional: true,
        },
    ])
}

fn cron_tick_event_type() -> lashlang::NamedDataType {
    lashlang::NamedDataType::object(
        "cron.Tick",
        vec![field("fired_at", lashlang::TypeExpr::Str)],
    )
    .expect("valid cron tick type")
}

fn button_trigger_event_type() -> lashlang::NamedDataType {
    lashlang::NamedDataType::object(
        "ui.button.Pressed",
        vec![
            field(
                "button",
                lashlang::TypeExpr::Union(vec![
                    lashlang::TypeExpr::Enum(vec!["Red".into()]),
                    lashlang::TypeExpr::Enum(vec!["Blue".into()]),
                ]),
            ),
            field("message", lashlang::TypeExpr::Str),
            field("pressed_at", lashlang::TypeExpr::Str),
        ],
    )
    .expect("valid button trigger event type")
}

fn button_trigger_payload_schema() -> lash::triggers::LashSchema {
    lash::triggers::LashSchema::new(serde_json::json!({
        "type": "object",
        "properties": {
            "button": { "type": "string", "enum": ["Red", "Blue"] },
            "message": { "type": "string" },
            "pressed_at": { "type": "string" }
        },
        "required": ["button", "message", "pressed_at"],
        "additionalProperties": false
    }))
}

fn mail_received_event_type() -> lashlang::NamedDataType {
    lashlang::NamedDataType::object(
        "mail.Received",
        vec![
            field("account", lashlang::TypeExpr::Str),
            field("title", lashlang::TypeExpr::Str),
            field("text", lashlang::TypeExpr::Str),
        ],
    )
    .expect("valid mail received event type")
}

fn workbench_lashlang_resources() -> lashlang::LashlangHostCatalog {
    let mut resources = lashlang::LashlangHostCatalog::new();
    resources
        .add_trigger_source_constructor(
            ["cron", "Schedule"],
            schedule_config_type(),
            cron_tick_event_type(),
        )
        .expect("valid cron trigger source");
    resources
        .add_trigger_source_constructor(
            ["mail", "received"],
            lashlang::TypeExpr::Object(vec![]),
            mail_received_event_type(),
        )
        .expect("valid mail trigger source");
    resources
}

fn declare_button_event(reg: &mut PluginRegistrar) -> Result<(), PluginError> {
    reg.triggers().declare(TriggerEvent::new(
        "Button",
        "ui.button",
        "pressed",
        button_trigger_payload_schema(),
    ))?;
    Ok(())
}
// docs:end:workbench-resources
