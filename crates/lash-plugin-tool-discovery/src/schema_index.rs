#[cfg(feature = "semantic-tool-search")]
use crate::catalog::CatalogTool;

#[cfg(feature = "semantic-tool-search")]
pub(crate) fn semantic_index_text(tool: &CatalogTool) -> String {
    let mut parts = vec![
        tool.contract.name.clone(),
        tool.contract.render_signature(),
        tool.contract.description.clone(),
    ];
    let return_details = tool.contract.render_returns();
    if !return_details.is_empty() {
        parts.push(return_details);
    }
    parts.extend(tool.contract.examples.clone());
    parts
        .into_iter()
        .map(|part| part.trim().to_string())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}
