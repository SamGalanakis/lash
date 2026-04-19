use crate::llm::types::LlmToolSpec;
use crate::session_model::model_tool_specs_iter;
use crate::{ExecutionMode, ToolDefinition};

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
    pub enabled: Option<bool>,
    pub injected: Option<bool>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ToolSurface {
    pub tools: Vec<ToolDefinition>,
    pub tool_list_notes: Vec<String>,
}

impl ToolSurface {
    pub fn from_tools(tools: Vec<ToolDefinition>) -> Self {
        Self {
            tools,
            tool_list_notes: Vec::new(),
        }
    }

    pub fn enabled_tools_iter(&self) -> impl Iterator<Item = &ToolDefinition> {
        self.tools.iter().filter(|tool| tool.enabled)
    }

    pub fn enabled_tools(&self) -> Vec<ToolDefinition> {
        self.enabled_tools_iter().cloned().collect()
    }

    pub fn prompt_tools_iter(&self) -> impl Iterator<Item = &ToolDefinition> {
        self.tools
            .iter()
            .filter(|tool| tool.enabled && tool.injected)
    }

    pub fn prompt_tools(&self) -> Vec<ToolDefinition> {
        self.prompt_tools_iter().cloned().collect()
    }

    pub fn has_enabled_tool(&self, tool_name: &str) -> bool {
        self.tools
            .iter()
            .any(|tool| tool.enabled && tool.name == tool_name)
    }

    pub fn tool_names(&self) -> Vec<String> {
        self.enabled_tools_iter()
            .map(|tool| tool.name.clone())
            .collect()
    }

    pub fn omitted_tool_count(&self) -> usize {
        self.tools
            .iter()
            .filter(|tool| tool.enabled)
            .filter(|tool| !tool.injected)
            .count()
    }

    pub fn model_tool_specs(&self) -> Vec<LlmToolSpec> {
        model_tool_specs_iter(self.enabled_tools_iter())
    }

    pub fn prompt_tool_docs(&self) -> String {
        let mut docs = ToolDefinition::format_tool_docs_iter(self.prompt_tools_iter());
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
}

pub fn build_tool_surface(input: ToolSurfaceBuildInput) -> ToolSurface {
    let mut surface = ToolSurface::from_tools(input.tools);
    if matches!(input.mode, ExecutionMode::Rlm) {
        apply_rlm_surface_rules(&mut surface);
    }
    for contribution in input.contributions {
        apply_contribution(&mut surface, contribution);
    }
    surface
}

fn apply_rlm_surface_rules(surface: &mut ToolSurface) {
    let omitted_tool_count = surface
        .tools
        .iter()
        .filter(|tool| tool.enabled)
        .filter(|tool| tool.name != "search_tools")
        .filter(|tool| !tool.injected)
        .count();

    if let Some(tool) = surface
        .tools
        .iter_mut()
        .find(|tool| tool.name == "search_tools")
    {
        let enabled = omitted_tool_count > 0;
        tool.enabled = enabled;
        tool.injected = enabled;
    }

    if omitted_tool_count > 0 {
        surface.tool_list_notes.push(format!(
            "- **Note:** {omitted_tool_count} additional tool(s) are available but omitted from this prompt for brevity."
        ));
    }
}

fn apply_contribution(surface: &mut ToolSurface, contribution: ToolSurfaceContribution) {
    for override_ in contribution.overrides {
        if let Some(tool) = surface
            .tools
            .iter_mut()
            .find(|tool| tool.name == override_.tool_name)
        {
            if let Some(enabled) = override_.enabled {
                tool.enabled = enabled;
            }
            if let Some(injected) = override_.injected {
                tool.injected = injected;
            }
            if !tool.enabled {
                tool.injected = false;
            }
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
    use crate::{ToolExecutionMode, ToolParam, ToolSurfaceOverride};

    fn tool(name: &str, injected: bool) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: format!("Tool {name}"),
            params: vec![ToolParam::typed("path", "str")],
            returns: "str".to_string(),
            examples: Vec::new(),
            enabled: true,
            injected,
            input_schema_override: None,
            output_schema_override: None,
            execution_mode: ToolExecutionMode::Parallel,
        }
    }

    #[test]
    fn rlm_surface_enables_search_tool_only_when_tools_are_omitted() {
        let surface = build_tool_surface(ToolSurfaceBuildInput {
            tools: vec![
                tool("search_tools", false),
                tool("read_file", true),
                tool("grep", false),
            ],
            mode: crate::ExecutionMode::Rlm,
            contributions: Vec::new(),
        });

        let search_tools = surface
            .tools
            .iter()
            .find(|tool| tool.name == "search_tools")
            .expect("search_tools present");
        assert!(search_tools.enabled);
        assert!(search_tools.injected);
        assert_eq!(surface.omitted_tool_count(), 1);
        assert!(surface.prompt_tool_docs().contains("additional tool(s)"));
    }

    #[test]
    fn explicit_contributions_apply_after_builtin_rules() {
        let surface = build_tool_surface(ToolSurfaceBuildInput {
            tools: vec![tool("search_tools", false), tool("read_file", true)],
            mode: crate::ExecutionMode::Rlm,
            contributions: vec![ToolSurfaceContribution {
                overrides: vec![ToolSurfaceOverride {
                    tool_name: "read_file".to_string(),
                    enabled: Some(false),
                    injected: None,
                }],
                tool_list_notes: vec!["custom note".to_string()],
            }],
        });

        assert!(
            !surface
                .tools
                .iter()
                .find(|tool| tool.name == "read_file")
                .expect("read_file present")
                .enabled
        );
        assert!(
            surface
                .tool_list_notes
                .iter()
                .any(|note| note == "custom note")
        );
    }
}
