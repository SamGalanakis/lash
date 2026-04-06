use crate::{MessageRole, Part};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginMessage {
    pub role: MessageRole,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<Part>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<Vec<u8>>,
}

impl PluginMessage {
    pub fn text(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            parts: Vec::new(),
            images: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptContribution {
    pub section: crate::PromptSectionName,
    pub block: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default)]
    pub priority: i32,
    pub content: String,
}

impl PromptContribution {
    pub fn block(
        section: crate::PromptSectionName,
        block: impl Into<String>,
        title: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        let title = title.into();
        let title = (!title.trim().is_empty()).then_some(title);
        Self {
            section,
            block: block.into(),
            title,
            priority: 0,
            content: content.into(),
        }
    }

    pub fn guidance(
        block: impl Into<String>,
        title: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            section: crate::PromptSectionName::Guidance,
            ..Self::block(crate::PromptSectionName::Guidance, block, title, content)
        }
    }

    pub fn environment(
        block: impl Into<String>,
        title: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            section: crate::PromptSectionName::Environment,
            ..Self::block(crate::PromptSectionName::Environment, block, title, content)
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
