use crate::llm::types::LlmToolSpec;
use crate::session_model::model_tool_specs_iter;
use crate::{ExecutionMode, PromptContribution, ToolAvailability, ToolDefinition};

#[derive(Clone, Debug)]
pub struct ToolSurfaceBuildInput {
    pub tools: Vec<ToolDefinition>,
    pub mode: ExecutionMode,
    pub contributions: Vec<ToolSurfaceContribution>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ToolSurfaceContribution {
    pub overrides: Vec<ToolSurfaceOverride>,
    pub tool_list_notes: Vec<String>,
}

impl ToolSurfaceContribution {
    pub fn is_empty(&self) -> bool {
        self.overrides.is_empty() && self.tool_list_notes.is_empty()
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ToolSurfaceOverride {
    pub tool_name: String,
    pub availability: Option<ToolAvailability>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolSurfaceEntry {
    pub definition: ToolDefinition,
    pub availability: ToolAvailability,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ToolSurface {
    pub tools: Vec<ToolSurfaceEntry>,
    pub tool_list_notes: Vec<String>,
}

impl ToolSurface {
    pub fn from_tools(tools: Vec<ToolDefinition>, mode: ExecutionMode) -> Self {
        Self {
            tools: tools
                .into_iter()
                .map(|definition| ToolSurfaceEntry {
                    availability: definition.effective_availability(&mode),
                    definition,
                })
                .collect(),
            tool_list_notes: Vec::new(),
        }
    }

    pub fn callable_tools_iter(&self) -> impl Iterator<Item = &ToolDefinition> {
        self.tools
            .iter()
            .filter(|tool| tool.availability.is_callable())
            .map(|tool| &tool.definition)
    }

    pub fn callable_tools(&self) -> Vec<ToolDefinition> {
        self.callable_tools_iter().cloned().collect()
    }

    pub fn documented_tools_iter(&self) -> impl Iterator<Item = &ToolDefinition> {
        self.tools
            .iter()
            .filter(|tool| tool.availability.is_documented())
            .map(|tool| &tool.definition)
    }

    pub fn documented_tools(&self) -> Vec<ToolDefinition> {
        self.documented_tools_iter().cloned().collect()
    }

    pub fn discoverable_tools_iter(&self) -> impl Iterator<Item = &ToolSurfaceEntry> {
        self.tools
            .iter()
            .filter(|tool| tool.availability.is_discoverable())
    }

    pub fn has_callable_tool(&self, tool_name: &str) -> bool {
        self.tools
            .iter()
            .any(|tool| tool.availability.is_callable() && tool.definition.name == tool_name)
    }

    pub fn tool_availability(&self, tool_name: &str) -> Option<ToolAvailability> {
        self.tools
            .iter()
            .find(|tool| tool.definition.name == tool_name)
            .map(|tool| tool.availability)
    }

    pub fn tool_names(&self) -> Vec<String> {
        self.callable_tools_iter()
            .map(|tool| tool.name.clone())
            .collect()
    }

    pub fn omitted_tool_count(&self) -> usize {
        self.tools
            .iter()
            .filter(|tool| tool.availability.is_discoverable())
            .filter(|tool| !tool.availability.is_documented())
            .count()
    }

    pub fn model_tool_specs(&self) -> Vec<LlmToolSpec> {
        model_tool_specs_iter(self.callable_tools_iter())
    }

    pub fn prompt_tool_docs(&self) -> String {
        let mut docs = ToolDefinition::format_tool_docs_iter(self.documented_tools_iter());
        for note in &self.tool_list_notes {
            let note = note.trim();
            if note.is_empty() {
                continue;
            }
            if !docs.is_empty() {
                docs.push_str("\n\n");
            }
            docs.push_str(note);
        }
        docs
    }

    pub fn filter_prompt_contributions(
        &self,
        contributions: Vec<PromptContribution>,
    ) -> Vec<PromptContribution> {
        contributions
            .into_iter()
            .filter(|contribution| self.includes_prompt_contribution(contribution))
            .collect()
    }

    fn includes_prompt_contribution(&self, contribution: &PromptContribution) -> bool {
        if contribution.gate.is_empty() {
            return true;
        }
        contribution.gate.tools.iter().any(|tool_name| {
            self.tool_availability(tool_name)
                .is_some_and(|availability| availability >= contribution.gate.minimum_availability)
        })
    }
}

pub fn build_tool_surface(input: ToolSurfaceBuildInput) -> ToolSurface {
    let mut surface = ToolSurface::from_tools(input.tools, input.mode);
    for contribution in input.contributions {
        apply_contribution(&mut surface, contribution);
    }
    if surface.omitted_tool_count() > 0 {
        surface.tool_list_notes.push(format!(
            "- **Note:** {} additional discoverable tool(s) are available but omitted from Available Tools for brevity.",
            surface.omitted_tool_count()
        ));
    }
    surface
}

fn apply_contribution(surface: &mut ToolSurface, contribution: ToolSurfaceContribution) {
    for override_ in contribution.overrides {
        if let Some(tool) = surface
            .tools
            .iter_mut()
            .find(|tool| tool.definition.name == override_.tool_name)
            && let Some(availability) = override_.availability
        {
            tool.availability = availability;
        }
    }

    surface.tool_list_notes.extend(
        contribution
            .tool_list_notes
            .into_iter()
            .map(|note| note.trim().to_string())
            .filter(|note| !note.is_empty()),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToolActivation, ToolAvailabilityConfig, ToolExecutionMode, ToolParam};

    fn tool(name: &str, availability: ToolAvailability) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: format!("Tool {name}"),
            params: vec![ToolParam::typed("path", "str")],
            returns: "str".to_string(),
            examples: Vec::new(),
            availability: ToolAvailabilityConfig::same(availability),
            activation: ToolActivation::Always,
            availability_override: None,
            input_schema_override: None,
            output_schema_override: None,
            execution_mode: ToolExecutionMode::Parallel,
        }
    }

    #[test]
    fn surface_splits_callable_and_documented_tools() {
        let surface = build_tool_surface(ToolSurfaceBuildInput {
            tools: vec![
                tool("discover_tools", ToolAvailability::Documented),
                tool("read_file", ToolAvailability::Documented),
                tool("grep", ToolAvailability::Callable),
                tool("privileged_tool", ToolAvailability::Discoverable),
            ],
            mode: crate::ExecutionMode::new("test_mode"),
            contributions: Vec::new(),
        });

        assert_eq!(surface.callable_tools().len(), 3);
        assert_eq!(surface.documented_tools().len(), 2);
        assert_eq!(surface.omitted_tool_count(), 2);
        assert!(surface.prompt_tool_docs().contains("discoverable tool(s)"));
    }

    #[test]
    fn explicit_contributions_override_availability() {
        let surface = build_tool_surface(ToolSurfaceBuildInput {
            tools: vec![tool("read_file", ToolAvailability::Documented)],
            mode: crate::ExecutionMode::new("test_mode"),
            contributions: vec![ToolSurfaceContribution {
                overrides: vec![ToolSurfaceOverride {
                    tool_name: "read_file".to_string(),
                    availability: Some(ToolAvailability::Hidden),
                }],
                tool_list_notes: vec!["custom note".to_string()],
            }],
        });

        assert_eq!(
            surface
                .tools
                .iter()
                .find(|tool| tool.definition.name == "read_file")
                .expect("read_file present")
                .availability,
            ToolAvailability::Hidden
        );
        assert!(
            surface
                .tool_list_notes
                .iter()
                .any(|note| note == "custom note")
        );
    }

    #[test]
    fn prompt_gate_requires_matching_tool_availability() {
        let surface = build_tool_surface(ToolSurfaceBuildInput {
            tools: vec![tool("discover_tools", ToolAvailability::Documented)],
            mode: crate::ExecutionMode::standard(),
            contributions: Vec::new(),
        });

        let kept = surface.filter_prompt_contributions(vec![
            PromptContribution::guidance("Plain", "always"),
            PromptContribution::guidance("Discovery", "discover")
                .requires_tool("discover_tools", ToolAvailability::Documented),
            PromptContribution::guidance("Hidden", "hidden")
                .requires_tool("load_tools", ToolAvailability::Callable),
        ]);

        assert_eq!(kept.len(), 2);
        assert!(
            kept.iter()
                .any(|contribution| contribution.title.as_deref() == Some("Plain"))
        );
        assert!(
            kept.iter()
                .any(|contribution| contribution.title.as_deref() == Some("Discovery"))
        );
    }
}
