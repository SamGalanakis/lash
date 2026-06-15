use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use super::*;

fn merge_string_array(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    values: Vec<String>,
) {
    let mut existing = obj
        .remove(key)
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect::<BTreeSet<_>>();
    existing.extend(
        values
            .into_iter()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
    );
    if !existing.is_empty() {
        obj.insert(key.to_string(), serde_json::json!(existing));
    }
}

fn apply_tool_discovery_contributions(
    catalog: &mut [serde_json::Value],
    contributions: impl IntoIterator<Item = ToolDiscoveryContribution>,
) {
    let mut by_name = BTreeMap::new();
    for (idx, tool) in catalog.iter().enumerate() {
        if let Some(name) = tool.get("name").and_then(serde_json::Value::as_str) {
            by_name.insert(name.to_string(), idx);
        }
    }

    for contribution in contributions {
        for patch in contribution.tools {
            let Some(idx) = by_name.get(&patch.tool_name).copied() else {
                continue;
            };
            let Some(obj) = catalog[idx].as_object_mut() else {
                continue;
            };
            if let Some(namespace) = patch
                .namespace
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
            {
                obj.insert("namespace".to_string(), serde_json::json!(namespace));
            }
            merge_string_array(obj, "aliases", patch.aliases);
        }
    }
}

impl PluginSession {
    pub fn resolved_tool_catalog(
        &self,
        session_id: &str,
    ) -> Result<Arc<crate::ToolCatalog>, PluginError> {
        let tools = self.tools.tool_manifests();
        let contract_provider = Arc::clone(&self.tools);
        let resolve_contract: lash_sansio::ToolContractResolver =
            Arc::new(move |name: &str| contract_provider.resolve_contract(name));
        Ok(Arc::new(self.resolve_tool_catalog(ToolCatalogContext {
            session_id: session_id.to_string(),
            tools,
            resolve_contract: Some(Arc::clone(&resolve_contract)),
            tool_access: self.tool_access.clone(),
            subagent: self.subagent.clone(),
            lashlang_abilities: self.lashlang_abilities,
        })?))
    }

    pub fn tool_catalog(&self, session_id: &str) -> Result<Vec<serde_json::Value>, PluginError> {
        let catalog = self.resolved_tool_catalog(session_id)?;
        let mut catalog =
            crate::tool_registry::project_tool_catalog(catalog.searchable_tools_iter().cloned());
        let contributions = collect_owned_sync(
            &self.contributions.tool_discovery_contributors,
            ToolDiscoveryContext {
                session_id: session_id.to_string(),
                catalog: catalog.clone(),
            },
            |hook, ctx| hook(ctx),
        )
        .unwrap_or_else(|err| {
            tracing::warn!("failed to resolve tool discovery metadata: {err}");
            Vec::new()
        });
        apply_tool_discovery_contributions(
            &mut catalog,
            contributions.into_iter().map(|owned| owned.value),
        );
        Ok(catalog)
    }

    pub fn resolve_tool_catalog(
        &self,
        ctx: ToolCatalogContext,
    ) -> Result<crate::ToolCatalog, PluginError> {
        let mut contributions = collect_owned_sync(
            &self.contributions.tool_catalog_contributors,
            ToolCatalogContext {
                session_id: ctx.session_id.clone(),
                tools: ctx.tools.clone(),
                resolve_contract: ctx.resolve_contract.clone(),
                tool_access: ctx.tool_access.clone(),
                subagent: ctx.subagent.clone(),
                lashlang_abilities: ctx.lashlang_abilities,
            },
            |hook, ctx| hook(ctx),
        )?
        .into_iter()
        .map(|owned| owned.value)
        .collect::<Vec<_>>();
        contributions.push(self.tool_catalog_overlay.clone());
        let (tools, resolve_contract) = if ctx.tool_access.tools.is_empty() {
            (ctx.tools, ctx.resolve_contract)
        } else {
            let contracts = ctx
                .tool_access
                .tools
                .iter()
                .map(|tool| (tool.name().to_string(), Arc::new(tool.contract())))
                .collect::<BTreeMap<_, _>>();
            (
                ctx.tool_access
                    .tools
                    .iter()
                    .map(|tool| tool.manifest())
                    .collect(),
                Some(Arc::new(move |name: &str| contracts.get(name).cloned())
                    as lash_sansio::ToolContractResolver),
            )
        };
        let authority_hidden_tools = tools
            .iter()
            .filter(|tool| ctx.tool_access.hides(&tool.name))
            .map(|tool| tool.name.clone())
            .collect::<BTreeSet<_>>();
        if !authority_hidden_tools.is_empty() {
            contributions.push(ToolCatalogContribution {
                overrides: authority_hidden_tools
                    .into_iter()
                    .map(|tool_name| ToolCatalogOverride {
                        tool_name,
                        availability: Some(crate::ToolAvailability::Off),
                    })
                    .collect(),
                ..Default::default()
            });
        }
        Ok(crate::build_tool_catalog(crate::ToolCatalogBuildInput {
            tools,
            resolve_contract,
            contributions,
        }))
    }
}
