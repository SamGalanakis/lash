use std::sync::Arc;

use lash_core::{ExecutionMode, PluginHost};

fn tool_names(session: &lash_core::PluginSession) -> Vec<String> {
    session
        .tool_surface("root", ExecutionMode::standard())
        .expect("tool surface")
        .tool_names()
        .as_ref()
        .clone()
}

#[test]
fn standard_mode_owns_batch_not_process_controls() {
    let session = PluginHost::new(vec![Arc::new(
        lash_mode_standard::BuiltinStandardModePluginFactory,
    )])
    .build_standard_session("root", None)
    .expect("session");

    let names = tool_names(&session);
    assert!(names.contains(&"batch".to_string()));
    assert!(!names.contains(&"list_process_handles".to_string()));
    assert!(!names.contains(&"cancel_process".to_string()));
}

#[test]
fn process_controls_are_composed_with_standard_mode() {
    let session = PluginHost::new(vec![
        Arc::new(lash_plugin_process_controls::ProcessControlsPluginFactory::new()),
        Arc::new(lash_tool_shell::StandardShellPluginFactory::new()),
        Arc::new(lash_mode_standard::BuiltinStandardModePluginFactory),
    ])
    .build_standard_session("root", None)
    .expect("session");

    let names = tool_names(&session);
    assert!(names.contains(&"batch".to_string()));
    assert!(names.contains(&"list_process_handles".to_string()));
    assert!(names.contains(&"cancel_process".to_string()));
}
