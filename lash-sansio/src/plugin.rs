use crate::{MessageRole, Part};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UserInputTransform {
    SkillBlockAppend {
        skill_name: String,
        skill_path: String,
    },
    LargePasteExpand {
        placeholder: String,
        expanded_char_count: usize,
        display_replacement: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserInputProvenance {
    pub display_text: String,
    pub effective_text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transforms: Vec<UserInputTransform>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginMessage {
    pub role: MessageRole,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<Part>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_input: Option<UserInputProvenance>,
}

impl PluginMessage {
    pub fn text(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            parts: Vec::new(),
            images: Vec::new(),
            user_input: None,
        }
    }

    pub fn user_input_provenance(&self) -> Option<&UserInputProvenance> {
        self.user_input.as_ref()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptContribution {
    pub slot: crate::PromptSlot,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default)]
    pub priority: i32,
    pub content: String,
}

impl PromptContribution {
    pub fn new(
        slot: crate::PromptSlot,
        title: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        let title = title.into();
        let title = (!title.trim().is_empty()).then_some(title);
        Self {
            slot,
            title,
            priority: 0,
            content: content.into(),
        }
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    pub fn intro(title: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(crate::PromptSlot::Intro, title, content)
    }

    pub fn execution(title: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(crate::PromptSlot::Execution, title, content)
    }

    pub fn guidance(title: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(crate::PromptSlot::Guidance, title, content)
    }

    pub fn project_instructions(content: impl Into<String>) -> Self {
        Self::new(
            crate::PromptSlot::ProjectInstructions,
            "Project Instructions",
            content,
        )
    }

    pub fn runtime_context(content: impl Into<String>) -> Self {
        Self::new(
            crate::PromptSlot::RuntimeContext,
            "Runtime Context",
            content,
        )
    }

    pub fn environment(title: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(crate::PromptSlot::Environment, title, content)
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
