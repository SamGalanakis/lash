use std::collections::BTreeMap;
use std::sync::Arc;

use crate::llm::types::LlmToolSpec;
use crate::{
    ExecutionMode, PromptContribution, ToolAvailability, ToolContract, ToolDefinition, ToolManifest,
};

pub type ToolContractResolver =
    Arc<dyn Fn(&str) -> Option<Arc<ToolContract>> + Send + Sync + 'static>;

#[derive(Clone)]
pub struct ToolSurfaceBuildInput {
    pub tools: Vec<ToolManifest>,
    pub mode: ExecutionMode,
    pub resolve_contract: Option<ToolContractResolver>,
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
    pub manifest: ToolManifest,
    pub availability: ToolAvailability,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ToolSurface {
    pub tools: Vec<ToolSurfaceEntry>,
    pub tool_list_notes: Vec<String>,
    #[serde(skip)]
    prompt_tool_docs: Arc<str>,
    #[serde(skip)]
    model_tool_specs: Arc<Vec<LlmToolSpec>>,
    #[serde(skip)]
    tool_names: Arc<Vec<String>>,
}

impl ToolSurface {
    pub fn from_tool_definitions(tools: Vec<ToolDefinition>, mode: ExecutionMode) -> Self {
        let contracts = tools
            .iter()
            .map(|tool| (tool.name.clone(), Arc::new(tool.contract())))
            .collect();
        Self::from_tools(
            tools.into_iter().map(|tool| tool.manifest()).collect(),
            mode,
            contracts,
        )
    }

    pub fn from_tools(
        tools: Vec<ToolManifest>,
        mode: ExecutionMode,
        contracts: BTreeMap<String, Arc<ToolContract>>,
    ) -> Self {
        let mut surface = Self::from_tool_manifests(tools, &mode);
        surface.rebuild_projections(&contracts);
        surface
    }

    fn from_tool_manifests(tools: Vec<ToolManifest>, mode: &ExecutionMode) -> Self {
        Self {
            tools: tools
                .into_iter()
                .map(|manifest| ToolSurfaceEntry {
                    availability: manifest.effective_availability(mode),
                    manifest,
                })
                .collect(),
            tool_list_notes: Vec::new(),
            prompt_tool_docs: Arc::from(""),
            model_tool_specs: Arc::new(Vec::new()),
            tool_names: Arc::new(Vec::new()),
        }
    }

    pub fn callable_tools_iter(&self) -> impl Iterator<Item = &ToolManifest> {
        self.tools
            .iter()
            .filter(|tool| tool.availability.is_callable())
            .map(|tool| &tool.manifest)
    }

    pub fn callable_tools(&self) -> Vec<ToolManifest> {
        self.callable_tools_iter().cloned().collect()
    }

    pub fn showcased_tools_iter(&self) -> impl Iterator<Item = &ToolManifest> {
        self.tools
            .iter()
            .filter(|tool| tool.availability.is_showcased())
            .map(|tool| &tool.manifest)
    }

    pub fn showcased_tools(&self) -> Vec<ToolManifest> {
        self.showcased_tools_iter().cloned().collect()
    }

    pub fn searchable_tools_iter(&self) -> impl Iterator<Item = &ToolSurfaceEntry> {
        self.tools
            .iter()
            .filter(|tool| tool.availability.is_searchable())
    }

    pub fn omitted_tools_iter(&self) -> impl Iterator<Item = &ToolSurfaceEntry> {
        self.searchable_tools_iter()
            .filter(|tool| !tool.availability.is_showcased())
    }

    pub fn has_callable_tool(&self, tool_name: &str) -> bool {
        self.tools
            .iter()
            .any(|tool| tool.availability.is_callable() && tool.manifest.name == tool_name)
    }

    pub fn tool_availability(&self, tool_name: &str) -> Option<ToolAvailability> {
        self.tools
            .iter()
            .find(|tool| tool.manifest.name == tool_name)
            .map(|tool| tool.availability)
    }

    pub fn tool_names(&self) -> Arc<Vec<String>> {
        Arc::clone(&self.tool_names)
    }

    pub fn omitted_tool_count(&self) -> usize {
        self.omitted_tools_iter().count()
    }

    pub fn model_tool_specs(&self) -> Arc<Vec<LlmToolSpec>> {
        Arc::clone(&self.model_tool_specs)
    }

    pub fn prompt_tool_docs(&self) -> &str {
        &self.prompt_tool_docs
    }

    fn rendered_prompt_tool_docs(&self, contracts: &BTreeMap<String, Arc<ToolContract>>) -> String {
        let mut docs = self
            .tools
            .iter()
            .filter(|tool| tool.availability.is_showcased())
            .filter_map(|tool| {
                contracts
                    .get(&tool.manifest.name)
                    .map(|contract| contract.compact_contract(&tool.manifest).render_markdown())
            })
            .collect::<Vec<_>>()
            .join("\n\n");
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

    fn rebuild_projections(&mut self, contracts: &BTreeMap<String, Arc<ToolContract>>) {
        self.prompt_tool_docs = Arc::from(self.rendered_prompt_tool_docs(contracts));
        self.model_tool_specs = Arc::new(
            self.tools
                .iter()
                .filter(|tool| tool.availability.is_callable())
                .filter_map(|tool| {
                    contracts
                        .get(&tool.manifest.name)
                        .map(|contract| contract.model_tool(&tool.manifest))
                })
                .map(|model_tool| LlmToolSpec {
                    name: model_tool.name,
                    description: model_tool.description,
                    input_schema: model_tool.input_schema,
                    output_schema: model_tool.output_schema,
                    input_schema_projections: model_tool.input_schema_projections,
                    output_schema_projections: model_tool.output_schema_projections,
                })
                .collect(),
        );
        self.tool_names = Arc::new(
            self.tools
                .iter()
                .filter(|tool| tool.availability.is_callable())
                .map(|tool| tool.manifest.name.clone())
                .collect(),
        );
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
    let mode = input.mode;
    let mut surface = ToolSurface::from_tool_manifests(input.tools, &mode);
    for contribution in input.contributions {
        apply_contribution(&mut surface, contribution);
    }
    let mut contracts = BTreeMap::new();
    if let Some(resolve_contract) = input.resolve_contract {
        for tool in &surface.tools {
            if !needs_projection_contract(&mode, tool.availability) {
                continue;
            }
            if let Some(contract) = resolve_contract(&tool.manifest.name) {
                contracts.insert(tool.manifest.name.clone(), contract);
            }
        }
    }
    surface.rebuild_projections(&contracts);
    surface
}

fn needs_projection_contract(mode: &ExecutionMode, availability: ToolAvailability) -> bool {
    availability.is_showcased() || (mode.plugin_id() != "rlm" && availability.is_callable())
}

fn apply_contribution(surface: &mut ToolSurface, contribution: ToolSurfaceContribution) {
    for override_ in contribution.overrides {
        if let Some(tool) = surface
            .tools
            .iter_mut()
            .find(|tool| tool.manifest.name == override_.tool_name)
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
    use crate::{ToolActivation, ToolAvailabilityConfig, ToolExecutionMode};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn tool(name: &str, availability: ToolAvailability) -> ToolDefinition {
        let mut definition = ToolDefinition::raw(
            name,
            format!("Tool {name}"),
            serde_json::json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
            serde_json::json!({ "type": "string" }),
        );
        definition.availability = ToolAvailabilityConfig::same(availability);
        definition.activation = ToolActivation::Always;
        definition.execution_mode = ToolExecutionMode::Parallel;
        definition
    }

    fn build_input(
        tools: Vec<ToolDefinition>,
        mode: crate::ExecutionMode,
        contributions: Vec<ToolSurfaceContribution>,
    ) -> ToolSurfaceBuildInput {
        let contracts = tools
            .iter()
            .map(|tool| (tool.name.clone(), Arc::new(tool.contract())))
            .collect::<BTreeMap<_, _>>();
        ToolSurfaceBuildInput {
            tools: tools.into_iter().map(|tool| tool.manifest()).collect(),
            mode,
            resolve_contract: Some(Arc::new(move |name| contracts.get(name).cloned())),
            contributions,
        }
    }

    #[test]
    fn surface_splits_callable_and_showcased_tools() {
        let surface = build_tool_surface(build_input(
            vec![
                tool("search_tools", ToolAvailability::Showcased),
                tool("read_file", ToolAvailability::Showcased),
                tool("grep", ToolAvailability::Callable),
                tool("privileged_tool", ToolAvailability::Searchable),
            ],
            crate::ExecutionMode::new("test_mode"),
            Vec::new(),
        ));

        assert_eq!(surface.callable_tools().len(), 3);
        assert_eq!(surface.showcased_tools().len(), 2);
        assert_eq!(surface.omitted_tool_count(), 2);
        assert!(!surface.prompt_tool_docs().contains("Catalogued tools"));
    }

    #[test]
    fn explicit_contributions_override_availability() {
        let surface = build_tool_surface(build_input(
            vec![tool("read_file", ToolAvailability::Showcased)],
            crate::ExecutionMode::new("test_mode"),
            vec![ToolSurfaceContribution {
                overrides: vec![ToolSurfaceOverride {
                    tool_name: "read_file".to_string(),
                    availability: Some(ToolAvailability::Off),
                }],
                tool_list_notes: vec!["custom note".to_string()],
            }],
        ));

        assert_eq!(
            surface
                .tools
                .iter()
                .find(|tool| tool.manifest.name == "read_file")
                .expect("read_file present")
                .availability,
            ToolAvailability::Off
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
        let surface = build_tool_surface(build_input(
            vec![tool("search_tools", ToolAvailability::Showcased)],
            crate::ExecutionMode::standard(),
            Vec::new(),
        ));

        let kept = surface.filter_prompt_contributions(vec![
            PromptContribution::guidance("Plain", "always"),
            PromptContribution::guidance("Discovery", "discover")
                .requires_tool("search_tools", ToolAvailability::Showcased),
            PromptContribution::guidance("Off", "off")
                .requires_tool("missing_tool", ToolAvailability::Callable),
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

    #[test]
    fn rlm_surface_does_not_resolve_searchable_only_contracts() {
        let contract_resolutions = Arc::new(AtomicUsize::new(0));
        let searchable = tool("large_schema", ToolAvailability::Searchable);
        let showcased = tool("search_tools", ToolAvailability::Showcased);
        let resolver_count = Arc::clone(&contract_resolutions);
        let surface = build_tool_surface(ToolSurfaceBuildInput {
            tools: vec![searchable.manifest(), showcased.manifest()],
            mode: crate::ExecutionMode::new("rlm"),
            resolve_contract: Some(Arc::new(move |name| {
                resolver_count.fetch_add(1, Ordering::SeqCst);
                match name {
                    "large_schema" => Some(Arc::new(searchable.contract())),
                    "search_tools" => Some(Arc::new(showcased.contract())),
                    _ => None,
                }
            })),
            contributions: Vec::new(),
        });

        assert_eq!(
            surface.tool_availability("large_schema"),
            Some(ToolAvailability::Searchable)
        );
        assert_eq!(contract_resolutions.load(Ordering::SeqCst), 1);
        assert!(!surface.prompt_tool_docs().contains("large_schema"));
    }
}
