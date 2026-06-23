//! Reference [`DeferredToolResolver`] for MCP call-paths.
//!
//! When the model calls a Lashlang module path that is not resident in the
//! link-time host environment, RLM linking asks this resolver to resolve it.
//! The resolver maps the call-path to a previously enumerated MCP tool and
//! returns a [`ToolGrant`] carrying the callable contract, the Lashlang
//! identity, and a Tool Execution Binding that records how to route the call
//! back to the originating MCP server. It resolves on demand only — it does not
//! enumerate, advertise, or rank.

use std::collections::HashMap;

use async_trait::async_trait;
use lash_core::ToolDefinition;
use lash_lashlang_runtime::{
    DeferredToolResolver, Resolution, ToolGrant, required_tool_lashlang_executable,
};
use serde_json::json;

/// One enumerated MCP tool plus the server it came from.
#[derive(Clone, Debug)]
pub struct McpCatalogedTool {
    pub server: String,
    pub definition: ToolDefinition,
}

/// Resolves deferred Lashlang call-paths to MCP tool grants. Built from the
/// MCP tools enumerated at session start (via `lash-plugin-mcp`).
pub struct McpDeferredToolResolver {
    by_call_path: HashMap<String, McpCatalogedTool>,
}

impl McpDeferredToolResolver {
    /// Build the resolver from the enumerated MCP tools, keyed by their
    /// Lashlang call-path. Tools without an explicit `lashlang.tool` binding
    /// are skipped (they cannot be called by module path under RLM).
    pub fn new(tools: impl IntoIterator<Item = McpCatalogedTool>) -> Self {
        let mut by_call_path = HashMap::new();
        for tool in tools {
            if let Ok(executable) = required_tool_lashlang_executable(&tool.definition.manifest) {
                by_call_path.insert(executable.call_path(), tool);
            }
        }
        Self { by_call_path }
    }
}

#[async_trait]
impl DeferredToolResolver for McpDeferredToolResolver {
    async fn resolve(&self, path: &str) -> Resolution {
        match self.by_call_path.get(path) {
            Some(tool) => {
                // The Tool Execution Binding records the routing the host needs
                // to fulfil and to rebuild the grant on replay/recovery: which
                // MCP server backs this tool and which underlying tool id to
                // call.
                let execution_binding = json!({
                    "kind": "mcp",
                    "server": tool.server,
                    "tool_id": tool.definition.manifest.id.to_string(),
                });
                Resolution::Resolved(
                    ToolGrant::new(tool.definition.clone())
                        .with_execution_binding(execution_binding),
                )
            }
            None => Resolution::NotAvailable,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::ToolContract;
    use lash_tool_support::{LashlangToolBinding, ToolDefinitionLashlangExt};

    fn mcp_tool(server: &str, name: &str, module: &str, operation: &str) -> McpCatalogedTool {
        let definition = ToolDefinition::raw(
            format!("tool:{server}/{name}"),
            name,
            format!("MCP tool {name}"),
            ToolContract::default_input_schema(),
            serde_json::json!({ "type": "object" }),
        )
        .with_lashlang_binding(LashlangToolBinding::new([module], operation));
        McpCatalogedTool {
            server: server.to_string(),
            definition,
        }
    }

    #[tokio::test]
    async fn resolves_known_call_path_with_execution_binding() {
        let resolver = McpDeferredToolResolver::new([mcp_tool(
            "appworld",
            "venmo_send",
            "venmo",
            "send",
        )]);
        let resolution = resolver.resolve("venmo.send").await;
        let Resolution::Resolved(grant) = resolution else {
            panic!("expected resolved grant");
        };
        assert_eq!(grant.execution_binding["kind"], json!("mcp"));
        assert_eq!(grant.execution_binding["server"], json!("appworld"));
        assert_eq!(grant.definition.name(), "venmo_send");
    }

    #[tokio::test]
    async fn unknown_call_path_is_not_available() {
        let resolver = McpDeferredToolResolver::new(std::iter::empty());
        assert!(matches!(
            resolver.resolve("mystery.run").await,
            Resolution::NotAvailable
        ));
    }
}
