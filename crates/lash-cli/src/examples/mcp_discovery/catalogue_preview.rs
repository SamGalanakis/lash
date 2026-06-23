//! Catalogue-preview prompt contribution.
//!
//! The flat Tool Catalog renders every resident member as a full prompt doc.
//! The long tail of MCP tools is intentionally *not* resident — it is deferred.
//! This formatter advertises that deferred tail to the model as an ordinary
//! [`PromptContribution`]: a compact module index plus the instruction to use
//! `tools.search(...)` and call the returned module path directly. It is host
//! policy delivered through a prompt contribution, not a catalog property.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use lash_core::PromptContribution;
use serde_json::Value;

use super::catalog::CatalogTool;

const CATALOGUE_MODULE_LIMIT: usize = 100;
const CATALOGUE_TOOL_NAME_LIMIT: usize = 50;

/// Build a catalogue-preview prompt contribution from the projected catalog of
/// deferred MCP tools, or `None` when there is nothing to advertise.
pub fn catalogue_preview_contribution(catalog: &[Value]) -> Option<PromptContribution> {
    let mut by_module: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut catalogued_count = 0usize;
    for tool in catalog.iter().filter_map(CatalogTool::from_value) {
        catalogued_count += 1;
        by_module
            .entry(tool.module_path.join("."))
            .or_default()
            .push(tool.call.clone());
    }
    if catalogued_count == 0 {
        return None;
    }
    for names in by_module.values_mut() {
        names.sort_unstable();
    }

    let mut rendered = format!(
        "Catalogued capabilities: {catalogued_count} capabilities are searchable through `tools.search(...)`.\n\
         When a task needs a capability not documented here, run `await tools.search({{ query: \"...\" }})?` and call the returned module path directly. \
         Results use the same compact contract shape as resident capabilities: call path, signature, description, and capped examples."
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

    if catalogued_count <= CATALOGUE_TOOL_NAME_LIMIT {
        rendered.push_str("\n\nCatalogued calls:");
        for (module, names) in by_module {
            rendered.push('\n');
            let _ = write!(rendered, "{module}: {}", names.join(", "));
        }
    }

    Some(
        PromptContribution::execution("Catalogued Capabilities", rendered)
            .requires_tool("search_tools"),
    )
}
