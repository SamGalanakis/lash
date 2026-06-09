//! Compiled sources for the Rust snippets on `docs/example-agent-workbench.html`.

use lash::host_events::HostEvent;
use lash::plugins::{PluginError, PluginRegistrar};

fn schedule_config_type() -> lashlang::TypeExpr {
    lashlang::TypeExpr::Object(vec![])
}

fn cron_tick_event_type() -> lashlang::NamedDataType {
    lashlang::NamedDataType::new("cron.Tick", lashlang::TypeExpr::Object(vec![]))
        .expect("valid event type")
}

fn button_trigger_event_type() -> lashlang::NamedDataType {
    lashlang::NamedDataType::new("ui.button.Pressed", lashlang::TypeExpr::Object(vec![]))
        .expect("valid event type")
}

// docs:start:workbench-resources
fn workbench_lashlang_resources() -> lashlang::ResourceCatalog {
    let mut resources = lashlang::ResourceCatalog::new();
    resources
        .add_trigger_source_constructor(
            ["cron", "Schedule"],
            schedule_config_type(),
            cron_tick_event_type(),
        )
        .expect("valid cron trigger source");
    resources
        .add_trigger_source_constructor(
            ["ui", "button", "pressed"],
            lashlang::TypeExpr::Object(vec![]),
            button_trigger_event_type(),
        )
        .expect("valid button trigger source");
    resources
}

fn declare_button_event(reg: &mut PluginRegistrar) -> Result<(), PluginError> {
    reg.host_events().declare(HostEvent::new(
        "Button",
        "ui.button",
        "pressed",
        button_trigger_event_type(),
    ))?;
    Ok(())
}
// docs:end:workbench-resources
