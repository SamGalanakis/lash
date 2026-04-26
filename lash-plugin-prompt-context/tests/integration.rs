use std::sync::Arc;

use lash::instructions::InstructionSource;
use lash::plugin::PromptHookContext;
use lash::testing::{MockSessionManager, mock_assembled_turn};
use lash::{
    ExecutionMode, PersistedSessionState, PluginHost, PromptSlot, SessionPolicy, SessionReadView,
    SessionSnapshot, SessionStateEnvelope,
};
use lash_plugin_prompt_context::{PromptContextPluginConfig, PromptContextPluginFactory};

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
    let mut factories = lash::testing::test_mode_factories();
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
            state: SessionReadView::new(SessionStateEnvelope::default()),
            mode_turn_options: lash::ModeTurnOptions::default(),
        })
        .await
        .expect("prompt contributions");

    assert!(contributions.iter().any(|contribution| {
        contribution.slot == PromptSlot::RuntimeContext
            && contribution.content.contains("Working directory:")
    }));
    assert!(contributions.iter().any(|contribution| {
        contribution.slot == PromptSlot::RuntimeContext
            && contribution.content.contains("Current date (UTC):")
    }));
    assert!(contributions.iter().any(|contribution| {
        contribution.slot == PromptSlot::ProjectInstructions
            && contribution.title.as_deref() == Some("Project Instructions")
            && contribution.content == "Repo rules"
    }));
}
