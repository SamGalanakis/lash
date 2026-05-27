use lash_core::plugin::AssistantProseProjectorPlugin;

pub(super) struct RlmAssistantProseProjector;

impl AssistantProseProjectorPlugin for RlmAssistantProseProjector {
    fn project_assistant_prose(&self, text: &str) -> String {
        crate::protocol::project_visible_assistant_prose(text)
    }
}
