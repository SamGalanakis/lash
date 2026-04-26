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

    pub fn first_text(&self) -> Option<&str> {
        if !self.content.is_empty() {
            return Some(self.content.as_str());
        }
        self.parts.iter().find_map(|part| {
            matches!(part.kind, crate::PartKind::Text | crate::PartKind::Prose)
                .then_some(part.content.as_str())
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptContributionGate {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<String>,
    #[serde(default)]
    pub minimum_availability: crate::ToolAvailability,
}

impl PromptContributionGate {
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptContribution {
    pub slot: crate::PromptSlot,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default)]
    pub priority: i32,
    #[serde(default, skip_serializing_if = "PromptContributionGate::is_empty")]
    pub gate: PromptContributionGate,
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
            gate: PromptContributionGate {
                tools: Vec::new(),
                minimum_availability: crate::ToolAvailability::default(),
            },
            content: content.into(),
        }
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    pub fn requires_tool(
        mut self,
        tool_name: impl Into<String>,
        minimum_availability: crate::ToolAvailability,
    ) -> Self {
        self.gate = PromptContributionGate {
            tools: vec![tool_name.into()],
            minimum_availability,
        };
        self
    }

    pub fn requires_any_tool(
        mut self,
        tool_names: impl IntoIterator<Item = impl Into<String>>,
        minimum_availability: crate::ToolAvailability,
    ) -> Self {
        self.gate = PromptContributionGate {
            tools: tool_names.into_iter().map(Into::into).collect(),
            minimum_availability,
        };
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
