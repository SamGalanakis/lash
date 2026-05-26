use std::collections::BTreeMap;
use std::fmt::Write as _;

use lash_core::plugin::{PluginError, ToolSurfaceContext};
use lash_core::{ToolAvailability, ToolSurfaceContribution, ToolSurfaceOverride};

pub(crate) fn rlm_tool_surface(
    ctx: ToolSurfaceContext,
) -> Result<ToolSurfaceContribution, PluginError> {
    if ctx.mode.plugin_id() != "rlm" {
        return Ok(ToolSurfaceContribution::default());
    }

    let has_catalogued_tools = has_catalogued_tools(&ctx);
    let overrides = ctx
        .tools
        .iter()
        .filter_map(|tool| {
            if tool.name == "search_tools" && !has_catalogued_tools {
                return Some(ToolSurfaceOverride {
                    tool_name: tool.name.clone(),
                    availability: Some(ToolAvailability::Off),
                });
            }
            let availability = tool.effective_availability(&ctx.mode);
            if availability == ToolAvailability::Searchable {
                Some(ToolSurfaceOverride {
                    tool_name: tool.name.clone(),
                    availability: Some(ToolAvailability::Callable),
                })
            } else {
                None
            }
        })
        .collect();

    Ok(ToolSurfaceContribution {
        overrides,
        tool_list_notes: catalogue_notes(&ctx, has_catalogued_tools),
    })
}

fn has_catalogued_tools(ctx: &ToolSurfaceContext) -> bool {
    ctx.tools.iter().any(|tool| {
        tool.name != "search_tools"
            && tool.effective_availability(&ctx.mode).is_searchable()
            && !tool.effective_availability(&ctx.mode).is_showcased()
    })
}

const CATALOGUE_NAMESPACE_LIMIT: usize = 100;
const CATALOGUE_TOOL_NAME_LIMIT: usize = 50;

fn catalogue_notes(ctx: &ToolSurfaceContext, has_catalogued_tools: bool) -> Vec<String> {
    if !has_catalogued_tools {
        return Vec::new();
    }

    let mut by_namespace: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    let mut omitted_tool_count = 0usize;
    for tool in &ctx.tools {
        if tool.name == "search_tools" {
            continue;
        }
        let availability = tool.effective_availability(&ctx.mode);
        if !availability.is_searchable() || availability.is_showcased() {
            continue;
        }
        omitted_tool_count += 1;
        let namespace = tool
            .discovery
            .namespace
            .as_deref()
            .filter(|namespace| !namespace.trim().is_empty())
            .unwrap_or("default");
        by_namespace
            .entry(namespace)
            .or_default()
            .push(tool.name.as_str());
    }
    for names in by_namespace.values_mut() {
        names.sort_unstable();
    }

    let mut rendered = format!(
        "Catalogued tools: {omitted_tool_count} other tools are searchable through `search_tools`.\n\
         When a task needs a tool not showcased here, run `search_tools(query=...)` and call the relevant result by name. \
         Results use the same compact contract shape as showcased tools: signature, description, and capped examples."
    );

    if by_namespace.len() <= CATALOGUE_NAMESPACE_LIMIT {
        rendered.push_str("\n\nNamespaces: ");
        for (index, (namespace, names)) in by_namespace.iter().enumerate() {
            if index > 0 {
                rendered.push_str(", ");
            }
            let _ = write!(rendered, "{namespace}({})", names.len());
        }
    } else {
        let _ = write!(
            rendered,
            "\n\nNamespaces: {} total; use `search_tools` to narrow them.",
            by_namespace.len()
        );
    }

    if omitted_tool_count <= CATALOGUE_TOOL_NAME_LIMIT {
        rendered.push_str("\n\nCatalogued names:");
        for (namespace, names) in by_namespace {
            rendered.push('\n');
            let _ = write!(rendered, "{namespace}: {}", names.join(", "));
        }
    }

    vec![rendered]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definitions::search_tools_definition;
    use lash_core::{
        ExecutionMode, ToolAvailabilityConfig, ToolContract, ToolDefinition, ToolExecutionMode,
        ToolSurfaceBuildInput, build_tool_surface,
    };
    use serde_json::json;
    use std::sync::Arc;

    #[test]
    fn rlm_surface_promotes_searchable_tools() {
        let tools = [
            search_tools_definition(),
            ToolDefinition::raw_named(
                "fetch_url",
                "Fetch URL",
                ToolContract::default_input_schema(),
                json!({ "type": "string" }),
            )
            .with_availability(ToolAvailabilityConfig::same(ToolAvailability::Searchable))
            .with_execution_mode(ToolExecutionMode::Parallel),
        ];
        let mode = ExecutionMode::new("rlm");
        let contracts: std::collections::BTreeMap<_, _> = tools
            .iter()
            .map(|tool| (tool.name.clone(), Arc::new(tool.contract())))
            .collect();
        let manifests = tools.iter().map(|tool| tool.manifest()).collect::<Vec<_>>();
        let contribution = rlm_tool_surface(ToolSurfaceContext {
            session_id: "session".to_string(),
            mode: mode.clone(),
            tools: manifests.clone(),
            resolve_contract: Some(Arc::new({
                let contracts = contracts.clone();
                move |name| contracts.get(name).cloned()
            })),
            tool_access: lash_core::SessionToolAccess::default(),
            subagent: None,
            lashlang_abilities: Default::default(),
        })
        .unwrap();
        let surface = build_tool_surface(ToolSurfaceBuildInput {
            tools: manifests,
            mode,
            resolve_contract: Some(Arc::new(move |name| contracts.get(name).cloned())),
            contributions: vec![contribution],
        });

        assert!(surface.has_callable_tool("fetch_url"));
        assert!(surface.prompt_tool_docs().contains("Catalogued tools:"));
    }

    #[test]
    fn rlm_surface_hides_search_when_no_catalogued_tools_exist() {
        let tool = search_tools_definition();
        let mode = ExecutionMode::new("rlm");
        let contribution = rlm_tool_surface(ToolSurfaceContext {
            session_id: "session".to_string(),
            mode,
            tools: vec![tool.manifest()],
            resolve_contract: None,
            tool_access: lash_core::SessionToolAccess::default(),
            subagent: None,
            lashlang_abilities: Default::default(),
        })
        .unwrap();

        assert_eq!(contribution.overrides.len(), 1);
        assert_eq!(contribution.overrides[0].tool_name, "search_tools");
        assert_eq!(
            contribution.overrides[0].availability,
            Some(ToolAvailability::Off)
        );
    }
}
