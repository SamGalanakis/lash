use std::sync::Arc;

use lash::{ExecutionMode, PluginHost};

fn tool_names(session: &lash::PluginSession) -> Vec<String> {
    session
        .tool_surface("root", ExecutionMode::standard())
        .tool_names()
}

#[test]
fn standard_mode_owns_batch_not_runtime_controls() {
    let session = PluginHost::new(vec![Arc::new(
        lash_mode_standard::BuiltinStandardModePluginFactory,
    )])
    .build_standard_session("root", None)
    .expect("session");

    let names = tool_names(&session);
    assert!(names.contains(&"batch".to_string()));
    assert!(!names.contains(&"monitor".to_string()));
    assert!(!names.contains(&"tasks_list".to_string()));
    assert!(!names.contains(&"tasks_stop".to_string()));
}

#[test]
fn runtime_controls_are_composed_with_standard_mode() {
    let session = PluginHost::new(vec![
        Arc::new(lash::BuiltinTaskControlsPluginFactory),
        Arc::new(lash::BuiltinMonitorToolPluginFactory),
        Arc::new(lash_mode_standard::BuiltinStandardModePluginFactory),
    ])
    .build_standard_session("root", None)
    .expect("session");

    let names = tool_names(&session);
    assert!(names.contains(&"batch".to_string()));
    assert!(names.contains(&"monitor".to_string()));
    assert!(names.contains(&"tasks_list".to_string()));
    assert!(names.contains(&"tasks_stop".to_string()));
}
