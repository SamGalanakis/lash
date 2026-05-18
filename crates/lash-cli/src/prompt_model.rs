#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PromptRequest {
    pub question: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub panel: Option<PromptPanel>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<String>,
    #[serde(default)]
    pub selection_mode: PromptSelectionMode,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub allow_note: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PromptPanel {
    pub title: String,
    pub markdown: String,
}

impl PromptRequest {
    pub(crate) fn freeform(question: impl Into<String>) -> Self {
        Self {
            question: question.into(),
            panel: None,
            options: Vec::new(),
            selection_mode: PromptSelectionMode::Single,
            allow_note: false,
        }
    }

    pub(crate) fn single(question: impl Into<String>, options: Vec<String>) -> Self {
        Self {
            question: question.into(),
            panel: None,
            options,
            selection_mode: PromptSelectionMode::Single,
            allow_note: false,
        }
    }

    pub(crate) fn multi(question: impl Into<String>, options: Vec<String>) -> Self {
        Self {
            question: question.into(),
            panel: None,
            options,
            selection_mode: PromptSelectionMode::Multi,
            allow_note: false,
        }
    }

    pub(crate) fn with_optional_note(mut self) -> Self {
        self.allow_note = !self.is_freeform();
        self
    }

    pub(crate) fn with_markdown_panel(
        mut self,
        title: impl Into<String>,
        markdown: impl Into<String>,
    ) -> Self {
        self.panel = Some(PromptPanel {
            title: title.into(),
            markdown: markdown.into(),
        });
        self
    }

    pub(crate) fn is_freeform(&self) -> bool {
        self.options.is_empty()
    }

    pub(crate) fn allows_note(&self) -> bool {
        self.allow_note && !self.is_freeform()
    }

    pub(crate) fn empty_response(&self) -> PromptResponse {
        if self.is_freeform() {
            PromptResponse::Text {
                text: String::new(),
            }
        } else {
            match self.selection_mode {
                PromptSelectionMode::Single => PromptResponse::Single {
                    selection: String::new(),
                    note: None,
                },
                PromptSelectionMode::Multi => PromptResponse::Multi {
                    selections: Vec::new(),
                    note: None,
                },
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PromptSelectionMode {
    #[default]
    Single,
    Multi,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum PromptResponse {
    Text {
        text: String,
    },
    Single {
        selection: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note: Option<String>,
    },
    Multi {
        selections: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note: Option<String>,
    },
}
