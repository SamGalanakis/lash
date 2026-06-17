use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use crate::llm::types::LlmToolSpec;
use crate::{
    PromptContribution, PromptFingerprint, ToolAvailability, ToolContract, ToolDefinition,
    ToolManifest, prompt_tool_names_fingerprint,
};

pub type ToolContractResolver =
    Arc<dyn Fn(&str) -> Option<Arc<ToolContract>> + Send + Sync + 'static>;

#[derive(Clone)]
pub struct ToolCatalogBuildInput {
    pub tools: Vec<ToolManifest>,
    pub resolve_contract: Option<ToolContractResolver>,
    pub contributions: Vec<ToolCatalogContribution>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ToolCatalogContribution {
    pub overrides: Vec<ToolCatalogOverride>,
    pub tool_list_notes: Vec<String>,
}

impl ToolCatalogContribution {
    pub fn is_empty(&self) -> bool {
        self.overrides.is_empty() && self.tool_list_notes.is_empty()
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ToolCatalogOverride {
    pub tool_name: String,
    pub availability: Option<ToolAvailability>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolCatalogEntry {
    pub manifest: ToolManifest,
    pub availability: ToolAvailability,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ToolCatalog {
    pub tools: Vec<ToolCatalogEntry>,
    pub tool_list_notes: Vec<String>,
    #[serde(skip)]
    resolve_contract: Option<ToolContractResolver>,
    #[serde(skip)]
    prompt_tool_docs: OnceLock<Arc<str>>,
    #[serde(skip)]
    model_tool_specs: OnceLock<Arc<Vec<LlmToolSpec>>>,
    #[serde(skip)]
    tool_names: OnceLock<Arc<Vec<String>>>,
    #[serde(skip)]
    tool_names_fingerprint: OnceLock<PromptFingerprint>,
}

impl Clone for ToolCatalog {
    fn clone(&self) -> Self {
        let clone = Self {
            tools: self.tools.clone(),
            tool_list_notes: self.tool_list_notes.clone(),
            resolve_contract: self.resolve_contract.clone(),
            prompt_tool_docs: OnceLock::new(),
            model_tool_specs: OnceLock::new(),
            tool_names: OnceLock::new(),
            tool_names_fingerprint: OnceLock::new(),
        };
        if let Some(value) = self.prompt_tool_docs.get() {
            let _ = clone.prompt_tool_docs.set(Arc::clone(value));
        }
        if let Some(value) = self.model_tool_specs.get() {
            let _ = clone.model_tool_specs.set(Arc::clone(value));
        }
        if let Some(value) = self.tool_names.get() {
            let _ = clone.tool_names.set(Arc::clone(value));
        }
        if let Some(value) = self.tool_names_fingerprint.get() {
            let _ = clone.tool_names_fingerprint.set(*value);
        }
        clone
    }
}

impl std::fmt::Debug for ToolCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolCatalog")
            .field("tools", &self.tools)
            .field("tool_list_notes", &self.tool_list_notes)
            .finish_non_exhaustive()
    }
}

impl Default for ToolCatalog {
    fn default() -> Self {
        Self {
            tools: Vec::new(),
            tool_list_notes: Vec::new(),
            resolve_contract: None,
            prompt_tool_docs: OnceLock::new(),
            model_tool_specs: OnceLock::new(),
            tool_names: OnceLock::new(),
            tool_names_fingerprint: OnceLock::new(),
        }
    }
}

impl ToolCatalog {
    pub fn from_tool_definitions(tools: Vec<ToolDefinition>) -> Self {
        let contracts = tools
            .iter()
            .map(|tool| (tool.name().to_string(), Arc::new(tool.contract())))
            .collect();
        Self::from_tools(
            tools.into_iter().map(|tool| tool.manifest()).collect(),
            contracts,
        )
    }

    pub fn from_tools(
        tools: Vec<ToolManifest>,
        contracts: BTreeMap<String, Arc<ToolContract>>,
    ) -> Self {
        let resolver_contracts = Arc::new(contracts);
        Self::from_tool_manifests(
            tools,
            Some(Arc::new(move |name| resolver_contracts.get(name).cloned())),
        )
    }

    fn from_tool_manifests(
        tools: Vec<ToolManifest>,
        resolve_contract: Option<ToolContractResolver>,
    ) -> Self {
        Self {
            tools: tools
                .into_iter()
                .map(|manifest| ToolCatalogEntry {
                    availability: manifest.effective_availability(),
                    manifest,
                })
                .collect(),
            tool_list_notes: Vec::new(),
            resolve_contract,
            prompt_tool_docs: OnceLock::new(),
            model_tool_specs: OnceLock::new(),
            tool_names: OnceLock::new(),
            tool_names_fingerprint: OnceLock::new(),
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

    pub fn searchable_tools_iter(&self) -> impl Iterator<Item = &ToolCatalogEntry> {
        self.tools
            .iter()
            .filter(|tool| tool.availability.is_searchable())
    }

    pub fn omitted_tools_iter(&self) -> impl Iterator<Item = &ToolCatalogEntry> {
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
        Arc::clone(self.tool_names.get_or_init(|| {
            Arc::new(
                self.tools
                    .iter()
                    .filter(|tool| tool.availability.is_callable())
                    .map(|tool| tool.manifest.name.clone())
                    .collect(),
            )
        }))
    }

    pub fn tool_names_fingerprint(&self) -> PromptFingerprint {
        *self
            .tool_names_fingerprint
            .get_or_init(|| prompt_tool_names_fingerprint(&self.tool_names()))
    }

    pub fn omitted_tool_count(&self) -> usize {
        self.omitted_tools_iter().count()
    }

    pub fn model_tool_specs(&self) -> Arc<Vec<LlmToolSpec>> {
        Arc::clone(self.model_tool_specs.get_or_init(|| {
            Arc::new(
                self.tools
                    .iter()
                    .filter(|tool| tool.availability.is_callable())
                    .filter_map(|tool| {
                        self.resolve_contract(&tool.manifest.name)
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
            )
        }))
    }

    pub fn prompt_tool_docs(&self) -> &str {
        self.prompt_tool_docs
            .get_or_init(|| Arc::from(self.rendered_prompt_tool_docs()))
            .as_ref()
    }

    fn resolve_contract(&self, tool_name: &str) -> Option<Arc<ToolContract>> {
        self.resolve_contract
            .as_ref()
            .and_then(|resolve| resolve(tool_name))
    }

    fn rendered_prompt_tool_docs(&self) -> String {
        let mut docs = self
            .tools
            .iter()
            .filter(|tool| tool.availability.is_showcased())
            .filter_map(|tool| {
                self.resolve_contract(&tool.manifest.name)
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

pub fn build_tool_catalog(input: ToolCatalogBuildInput) -> ToolCatalog {
    let mut surface = ToolCatalog::from_tool_manifests(input.tools, input.resolve_contract);
    for contribution in input.contributions {
        apply_contribution(&mut surface, contribution);
    }
    surface
}

fn apply_contribution(surface: &mut ToolCatalog, contribution: ToolCatalogContribution) {
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
    use crate::{ToolActivation, ToolAvailabilityConfig, ToolScheduling};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn tool(name: &str, availability: ToolAvailability) -> ToolDefinition {
        let mut definition = ToolDefinition::raw(
            format!("tool:{name}"),
            name,
            format!("Tool {name}"),
            serde_json::json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
            serde_json::json!({ "type": "string" }),
        );
        definition.manifest.availability = ToolAvailabilityConfig::same(availability);
        definition.manifest.activation = ToolActivation::Always;
        definition.manifest.scheduling = ToolScheduling::Parallel;
        definition
    }

    fn build_input(
        tools: Vec<ToolDefinition>,
        contributions: Vec<ToolCatalogContribution>,
    ) -> ToolCatalogBuildInput {
        let contracts = tools
            .iter()
            .map(|tool| (tool.name().to_string(), Arc::new(tool.contract())))
            .collect::<BTreeMap<_, _>>();
        ToolCatalogBuildInput {
            tools: tools.into_iter().map(|tool| tool.manifest()).collect(),
            resolve_contract: Some(Arc::new(move |name| contracts.get(name).cloned())),
            contributions,
        }
    }

    #[test]
    fn catalog_splits_callable_and_showcased_tools() {
        let surface = build_tool_catalog(build_input(
            vec![
                tool("search_tools", ToolAvailability::Showcased),
                tool("read_file", ToolAvailability::Showcased),
                tool("grep", ToolAvailability::Callable),
                tool("privileged_tool", ToolAvailability::Searchable),
            ],
            Vec::new(),
        ));

        assert_eq!(surface.callable_tools().len(), 3);
        assert_eq!(surface.showcased_tools().len(), 2);
        assert_eq!(surface.omitted_tool_count(), 2);
        assert!(!surface.prompt_tool_docs().contains("Catalogued tools"));
    }

    #[test]
    fn explicit_contributions_override_availability() {
        let surface = build_tool_catalog(build_input(
            vec![tool("read_file", ToolAvailability::Showcased)],
            vec![ToolCatalogContribution {
                overrides: vec![ToolCatalogOverride {
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
        let surface = build_tool_catalog(build_input(
            vec![tool("search_tools", ToolAvailability::Showcased)],
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
    fn rlm_catalog_does_not_resolve_searchable_only_contracts() {
        let contract_resolutions = Arc::new(AtomicUsize::new(0));
        let searchable = tool("large_schema", ToolAvailability::Searchable);
        let showcased = tool("search_tools", ToolAvailability::Showcased);
        let resolver_count = Arc::clone(&contract_resolutions);
        let surface = build_tool_catalog(ToolCatalogBuildInput {
            tools: vec![searchable.manifest(), showcased.manifest()],
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
        assert_eq!(contract_resolutions.load(Ordering::SeqCst), 0);
        assert!(!surface.prompt_tool_docs().contains("large_schema"));
        assert_eq!(contract_resolutions.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn callable_only_catalog_resolves_model_specs_lazily() {
        let contract_resolutions = Arc::new(AtomicUsize::new(0));
        let callable = tool("large_callable", ToolAvailability::Callable);
        let resolver_count = Arc::clone(&contract_resolutions);
        let surface = build_tool_catalog(ToolCatalogBuildInput {
            tools: vec![callable.manifest()],
            resolve_contract: Some(Arc::new(move |name| {
                resolver_count.fetch_add(1, Ordering::SeqCst);
                (name == "large_callable").then(|| Arc::new(callable.contract()))
            })),
            contributions: Vec::new(),
        });

        assert_eq!(
            surface.tool_names().as_ref(),
            &vec!["large_callable".to_string()]
        );
        assert_eq!(surface.model_tool_specs().len(), 1);
        assert_eq!(surface.prompt_tool_docs(), "");
        assert_eq!(contract_resolutions.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn standard_catalog_resolves_model_specs_lazily() {
        let contract_resolutions = Arc::new(AtomicUsize::new(0));
        let callable = tool("read_file", ToolAvailability::Callable);
        let resolver_count = Arc::clone(&contract_resolutions);
        let surface = build_tool_catalog(ToolCatalogBuildInput {
            tools: vec![callable.manifest()],
            resolve_contract: Some(Arc::new(move |name| {
                resolver_count.fetch_add(1, Ordering::SeqCst);
                (name == "read_file").then(|| Arc::new(callable.contract()))
            })),
            contributions: Vec::new(),
        });

        assert_eq!(contract_resolutions.load(Ordering::SeqCst), 0);
        assert_eq!(surface.model_tool_specs().len(), 1);
        assert_eq!(contract_resolutions.load(Ordering::SeqCst), 1);
        assert_eq!(surface.model_tool_specs().len(), 1);
        assert_eq!(contract_resolutions.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn tool_names_fingerprint_matches_prompt_hash() {
        let surface = build_tool_catalog(build_input(
            vec![
                tool("read_file", ToolAvailability::Callable),
                tool("search_tools", ToolAvailability::Showcased),
            ],
            Vec::new(),
        ));

        assert_eq!(
            surface.tool_names_fingerprint(),
            prompt_tool_names_fingerprint(&surface.tool_names())
        );
    }
}
