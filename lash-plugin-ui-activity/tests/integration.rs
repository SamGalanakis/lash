use std::sync::Arc;

use lash::plugin::RuntimeSessionHost;
use lash::testing::{MockSessionManager, mock_assembled_turn};
use lash::{
    ExecutionMode, PersistedSessionState, PluginDirective, PluginHost, PluginSurfaceEvent,
    SessionPolicy, SessionSnapshot, SessionStateEnvelope, TurnResultHookContext,
};
use lash_plugin_ui_activity::UiActivityPluginFactory;

fn mock_snapshot(run_session_id: &str) -> SessionSnapshot {
    PersistedSessionState::from_state(SessionStateEnvelope {
        session_id: "root".to_string(),
        policy: SessionPolicy {
            execution_mode: ExecutionMode::standard(),
            session_id: Some(run_session_id.to_string()),
            ..Default::default()
        },
        ..Default::default()
    })
}

fn mock_session_manager(run_session_id: &str) -> MockSessionManager {
    MockSessionManager::default()
        .with_snapshot(mock_snapshot(run_session_id))
        .with_turn(mock_assembled_turn(run_session_id, ""))
}

#[tokio::test]
async fn ui_activity_plugin_emits_done_notification_surface_event() {
    let host = {
        let mut f = lash::testing::test_mode_factories();
        f.push(Arc::new(UiActivityPluginFactory));
        PluginHost::new(f)
    };
    let session = host.build_standard_session("root", None).expect("session");
    let manager: Arc<dyn RuntimeSessionHost> = Arc::new(mock_session_manager("run-session"));

    let events = session
        .after_turn(TurnResultHookContext {
            session_id: "root".to_string(),
            turn: Arc::new(lash::TurnResultSummary::from_assembled(
                &mock_assembled_turn("root", ""),
            )),
            host: manager,
        })
        .await
        .expect("after turn");

    assert!(events.iter().any(|emitted| {
        matches!(
            &emitted.value,
            PluginDirective::EmitEvents { events }
                if events.iter().any(|event| matches!(
                    event,
                    PluginSurfaceEvent::Custom { name, payload }
                        if name == "desktop_notification"
                            && payload.get("body").and_then(|value| value.as_str())
                                == Some("Response complete")
                ))
        )
    }));
}
