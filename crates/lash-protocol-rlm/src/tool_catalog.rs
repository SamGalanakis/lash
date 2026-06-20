use std::collections::BTreeMap;
use std::fmt::Write as _;

use lash_core::plugin::{PluginError, ToolCatalogContext};
use lash_core::{ToolAvailability, ToolCatalog, ToolCatalogContribution, ToolCatalogOverride};
use lash_lashlang_runtime::required_tool_lashlang_executable;

pub(crate) fn rlm_tool_catalog(
    ctx: ToolCatalogContext,
) -> Result<ToolCatalogContribution, PluginError> {
    let has_catalogued_tools = has_catalogued_tools(&ctx);
    validate_rlm_lashlang_bindings(&ctx, has_catalogued_tools)?;
    let overrides = ctx
        .tools
        .iter()
        .filter_map(|tool| {
            if tool.name == "search_tools" && !has_catalogued_tools {
                return Some(ToolCatalogOverride {
                    tool_name: tool.name.clone(),
                    availability: Some(ToolAvailability::Off),
                });
            }
            let availability = tool.effective_availability();
            if availability == ToolAvailability::Searchable {
                Some(ToolCatalogOverride {
                    tool_name: tool.name.clone(),
                    availability: Some(ToolAvailability::Callable),
                })
            } else {
                None
            }
        })
        .collect();

    Ok(ToolCatalogContribution {
        overrides,
        tool_list_notes: catalogue_notes(&ctx, has_catalogued_tools),
    })
}

pub(crate) fn rlm_prompt_tool_docs(tool_catalog: &ToolCatalog) -> String {
    let mut docs = tool_catalog
        .tools
        .iter()
        .filter(|tool| tool.availability.is_showcased())
        .filter_map(|tool| {
            let contract = tool_catalog.resolve_contract(&tool.manifest.name)?;
            let binding = required_tool_lashlang_executable(&tool.manifest)
                .expect("RLM tool catalog registration must validate explicit Lashlang bindings");
            Some(
                contract
                    .compact_contract_with_signature_name(&tool.manifest, &binding.call_path())
                    .render_markdown(),
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    for note in &tool_catalog.tool_list_notes {
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

fn has_catalogued_tools(ctx: &ToolCatalogContext) -> bool {
    ctx.tools.iter().any(|tool| {
        tool.name != "search_tools"
            && tool.effective_availability().is_searchable()
            && !tool.effective_availability().is_showcased()
    })
}

fn validate_rlm_lashlang_bindings(
    ctx: &ToolCatalogContext,
    has_catalogued_tools: bool,
) -> Result<(), PluginError> {
    for tool in &ctx.tools {
        let availability = rlm_availability(tool, has_catalogued_tools);
        if availability.is_callable() {
            required_tool_lashlang_executable(tool).map_err(PluginError::Registration)?;
        }
    }
    Ok(())
}

fn rlm_availability(
    tool: &lash_core::ToolManifest,
    has_catalogued_tools: bool,
) -> ToolAvailability {
    if tool.name == "search_tools" && !has_catalogued_tools {
        return ToolAvailability::Off;
    }
    let availability = tool.effective_availability();
    if availability == ToolAvailability::Searchable {
        ToolAvailability::Callable
    } else {
        availability
    }
}

const CATALOGUE_MODULE_LIMIT: usize = 100;
const CATALOGUE_TOOL_NAME_LIMIT: usize = 50;

fn catalogue_notes(ctx: &ToolCatalogContext, has_catalogued_tools: bool) -> Vec<String> {
    if !has_catalogued_tools {
        return Vec::new();
    }

    let mut by_module: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut omitted_tool_count = 0usize;
    for tool in &ctx.tools {
        if tool.name == "search_tools" {
            continue;
        }
        let availability = rlm_availability(tool, has_catalogued_tools);
        if !availability.is_searchable() || availability.is_showcased() {
            continue;
        }
        omitted_tool_count += 1;
        let lashlang_binding = required_tool_lashlang_executable(tool)
            .expect("RLM tool catalog registration must validate explicit Lashlang bindings");
        let module = lashlang_binding.module_path.join(".");
        by_module
            .entry(module)
            .or_default()
            .push(lashlang_binding.call_path());
    }
    for names in by_module.values_mut() {
        names.sort_unstable();
    }

    let mut rendered = format!(
        "Catalogued capabilities: {omitted_tool_count} other capabilities are searchable through `tools.search(...)`.\n\
         When a task needs a capability not showcased here, run `await tools.search({{ query: \"...\" }})?` and call the returned module path directly. \
         Results use the same compact contract shape as showcased capabilities: call path, signature, description, and capped examples."
    );

    if by_module.len() <= CATALOGUE_MODULE_LIMIT {
        rendered.push_str("\n\nModules: ");
        for (index, (module, names)) in by_module.iter().enumerate() {
            if index > 0 {
                rendered.push_str(", ");
            }
            let _ = write!(rendered, "{module}({})", names.len());
        }
    } else {
        let _ = write!(
            rendered,
            "\n\nModules: {} total; use `tools.search` to narrow them.",
            by_module.len()
        );
    }

    if omitted_tool_count <= CATALOGUE_TOOL_NAME_LIMIT {
        rendered.push_str("\n\nCatalogued calls:");
        for (module, names) in by_module {
            rendered.push('\n');
            let _ = write!(rendered, "{module}: {}", names.join(", "));
        }
    }

    vec![rendered]
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::{
        ToolAvailabilityConfig, ToolCatalogBuildInput, ToolContract, ToolDefinition,
        ToolScheduling, build_tool_catalog,
    };
    use lash_lashlang_runtime::{LashlangSurface, LashlangToolBinding, ToolDefinitionLashlangExt};
    use serde_json::json;
    use std::sync::Arc;

    fn search_tools_definition() -> ToolDefinition {
        ToolDefinition::raw(
            "tool:test/search_tools",
            "search_tools",
            "Search tools",
            ToolContract::default_input_schema(),
            json!({ "type": "array" }),
        )
        .with_scheduling(ToolScheduling::Parallel)
        .with_lashlang_binding(LashlangToolBinding::new(["tools"], "search"))
    }

    #[test]
    fn rlm_catalog_promotes_searchable_tools() {
        let tools = [
            search_tools_definition(),
            ToolDefinition::raw(
                "tool:test/fetch_url",
                "fetch_url",
                "Fetch URL",
                ToolContract::default_input_schema(),
                json!({ "type": "string" }),
            )
            .with_availability(ToolAvailabilityConfig::same(ToolAvailability::Searchable))
            .with_scheduling(ToolScheduling::Parallel)
            .with_lashlang_binding(LashlangToolBinding::new(["web"], "fetch")),
        ];
        let contracts: std::collections::BTreeMap<_, _> = tools
            .iter()
            .map(|tool| (tool.name().to_string(), Arc::new(tool.contract())))
            .collect();
        let manifests = tools.iter().map(|tool| tool.manifest()).collect::<Vec<_>>();
        let contribution = rlm_tool_catalog(ToolCatalogContext {
            session_id: "session".to_string(),
            tools: manifests.clone(),
            resolve_contract: Some(Arc::new({
                let contracts = contracts.clone();
                move |name| contracts.get(name).cloned()
            })),
            tool_access: lash_core::SessionToolAccess::default(),
            subagent: None,
            extensions: Default::default(),
        })
        .unwrap();
        let surface = build_tool_catalog(ToolCatalogBuildInput {
            tools: manifests,
            resolve_contract: Some(Arc::new(move |name| contracts.get(name).cloned())),
            contributions: vec![contribution],
        });

        assert!(surface.has_callable_tool("fetch_url"));
        assert!(
            surface
                .prompt_tool_docs()
                .contains("Catalogued capabilities:")
        );
        let docs = rlm_prompt_tool_docs(&surface);
        assert!(docs.contains("web.fetch"));
        assert!(!docs.contains("fetch_url("));
    }

    #[test]
    fn rlm_catalog_rejects_callable_tools_without_lashlang_binding() {
        let missing = ToolDefinition::raw(
            "tool:test/update_plan",
            "update_plan",
            "Update plan",
            ToolContract::default_input_schema(),
            json!({ "type": "string" }),
        )
        .with_availability(ToolAvailabilityConfig::same(ToolAvailability::Callable))
        .with_scheduling(ToolScheduling::Parallel);

        let err = rlm_tool_catalog(ToolCatalogContext {
            session_id: "session".to_string(),
            tools: vec![missing.manifest()],
            resolve_contract: None,
            tool_access: lash_core::SessionToolAccess::default(),
            subagent: None,
            extensions: Default::default(),
        })
        .expect_err("missing binding should fail RLM registration");

        assert!(
            err.to_string()
                .contains("missing an explicit `lashlang.tool` binding"),
            "{err}"
        );
    }

    #[test]
    fn showcased_rlm_tool_docs_render_and_link_module_call() {
        let update_plan = ToolDefinition::raw(
            "tool:test/update_plan",
            "update_plan",
            "Update the visible plan",
            json!({
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": { "type": "string" },
                                "status": { "type": "string" }
                            },
                            "required": ["step", "status"]
                        }
                    }
                },
                "required": ["plan"],
                "additionalProperties": false
            }),
            json!({ "type": "string" }),
        )
        .with_availability(ToolAvailabilityConfig::showcased())
        .with_scheduling(ToolScheduling::Parallel)
        .with_lashlang_binding(LashlangToolBinding::new(["plan"], "update"));

        let contracts: std::collections::BTreeMap<_, _> = [update_plan.clone()]
            .iter()
            .map(|tool| (tool.name().to_string(), Arc::new(tool.contract())))
            .collect();
        let manifests = vec![update_plan.manifest()];
        let contribution = rlm_tool_catalog(ToolCatalogContext {
            session_id: "session".to_string(),
            tools: manifests.clone(),
            resolve_contract: Some(Arc::new({
                let contracts = contracts.clone();
                move |name| contracts.get(name).cloned()
            })),
            tool_access: lash_core::SessionToolAccess::default(),
            subagent: None,
            extensions: Default::default(),
        })
        .expect("RLM catalog validates explicit binding");
        let surface = build_tool_catalog(ToolCatalogBuildInput {
            tools: manifests,
            resolve_contract: Some(Arc::new(move |name| contracts.get(name).cloned())),
            contributions: vec![contribution],
        });

        let docs = rlm_prompt_tool_docs(&surface);
        assert!(docs.contains("plan.update("), "{docs}");
        assert!(!docs.contains("update_plan("), "{docs}");

        let host_environment = LashlangSurface::default()
            .host_environment(&surface)
            .expect("explicit binding builds host environment");
        let program = lashlang::parse(
            r#"await plan.update({ plan: [{ step: "Patch", status: "pending" }] })?"#,
        )
        .expect("module call parses");
        lashlang::LinkedModule::link(program, host_environment).expect("module call links");
    }

    #[test]
    fn rlm_catalog_hides_search_when_no_catalogued_tools_exist() {
        let tool = search_tools_definition();
        let contribution = rlm_tool_catalog(ToolCatalogContext {
            session_id: "session".to_string(),
            tools: vec![tool.manifest()],
            resolve_contract: None,
            tool_access: lash_core::SessionToolAccess::default(),
            subagent: None,
            extensions: Default::default(),
        })
        .unwrap();

        assert_eq!(contribution.overrides.len(), 1);
        assert_eq!(contribution.overrides[0].tool_name, "search_tools");
        assert_eq!(
            contribution.overrides[0].availability,
            Some(ToolAvailability::Off)
        );
        assert!(contribution.tool_list_notes.is_empty());
    }
}
