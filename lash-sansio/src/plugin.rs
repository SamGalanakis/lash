use crate::MessageRole;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginMessage {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptContribution {
    pub section: crate::PromptSectionName,
    #[serde(default)]
    pub priority: i32,
    pub content: String,
}

impl PromptContribution {
    pub fn guidance(content: impl Into<String>) -> Self {
        Self {
            section: crate::PromptSectionName::Guidance,
            priority: 0,
            content: content.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PluginSurfaceEvent {
    ModeIndicatorUpsert {
        key: String,
        label: String,
    },
    ModeIndicatorClear {
        key: String,
    },
    PanelUpsert {
        key: String,
        title: String,
        content: String,
    },
    PanelAppend {
        key: String,
        content: String,
    },
    PanelClear {
        key: String,
    },
    Custom {
        name: String,
        payload: serde_json::Value,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointKind {
    AfterWork,
    BeforeCompletion,
}
