use std::sync::Arc;

use lash_core::plugin::PromptHookContext;
use lash_core::testing::{MockSessionManager, mock_assembled_turn};
use lash_core::{
    ExecutionMode, PersistedSessionState, PluginHost, PromptSlot, SessionPolicy, SessionReadView,
    SessionSnapshot, SessionStateEnvelope,
};
use lash_plugin_prompt_context::{
    InstructionSource, PromptContextPluginConfig, PromptContextPluginFactory,
};

struct StaticInstructionSource {
    text: String,
    read_text: String,
}

impl InstructionSource for StaticInstructionSource {
    fn system_instructions(&self) -> String {
        self.text.clone()
    }

    fn context_instructions_for_reads(&self, _read_paths: &[String]) -> String {
        self.read_text.clone()
    }
}

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
async fn prompt_context_plugin_contributes_environment_and_project_instruction_sections() {
    let mut factories = lash_core::testing::test_mode_factories();
    factories.push(Arc::new(PromptContextPluginFactory::new(
        Arc::new(StaticInstructionSource {
            text: "Repo rules".to_string(),
            read_text: String::new(),
        }),
        PromptContextPluginConfig::default(),
    )));
    let host = PluginHost::new(factories);
    let session = host.build_standard_session("root", None).expect("session");
    let contributions = session
        .collect_prompt_contributions(PromptHookContext {
            session_id: "root".to_string(),
            host: Arc::new(mock_session_manager("run-session")),
            state: SessionReadView::from_exported_state(&SessionStateEnvelope::default()),
            mode_turn_options: lash_core::ModeTurnOptions::default(),
            turn_context: lash_core::TurnContext::default(),
        })
        .await
        .expect("prompt contributions");

    // Volatile environment (date, cwd, git state) is no longer a
    // PromptContribution — it now rides in a turn-prepare tail message
    // so the cached system prefix stays byte-stable across turns.
    assert!(
        !contributions
            .iter()
            .any(|contribution| contribution.slot == PromptSlot::RuntimeContext),
        "environment context should not appear as a runtime-context contribution",
    );
    assert!(contributions.iter().any(|contribution| {
        contribution.slot == PromptSlot::ProjectInstructions
            && contribution.title.as_deref() == Some("Project Instructions")
            && contribution.content.as_ref() == "Repo rules"
    }));
}
