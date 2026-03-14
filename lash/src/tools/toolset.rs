use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use crate::capabilities::{CapabilityId, tools_for_capability};
use crate::{
    ExecutionMode, ProgressSender, ToolDefinition, ToolPromptContext, ToolProvider, ToolResult,
};

/// Dependencies for constructing the default tool set.
#[derive(Default)]
pub struct ToolSetDeps {
    #[cfg(feature = "sqlite-store")]
    pub store: Option<Arc<crate::store::Store>>,
    pub tavily_api_key: Option<String>,
    pub skill_dirs: Option<Vec<PathBuf>>,
    pub prompt_bridge: Option<crate::PromptBridge>,
}

/// A composable set of tools with explicit composition methods.
///
/// ```rust,ignore
/// let tools = ToolSet::standard_defaults(deps)
///     .with_provider(my_custom_tool)
///     .without_tool("exec_command");
/// ```
pub struct ToolSet {
    tools: BTreeMap<String, (ToolDefinition, Arc<dyn ToolProvider>)>,
}

impl ToolSet {
    /// Create an empty tool set.
    pub fn new() -> Self {
        Self {
            tools: BTreeMap::new(),
        }
    }

    /// Core zero-dependency tools for a concrete execution mode.
    pub fn core_for(mode: ExecutionMode) -> Self {
        let shell_provider: Arc<dyn ToolProvider> = match mode {
            ExecutionMode::Standard => Arc::new(super::StandardShell::new()),
            ExecutionMode::Repl => Arc::new(super::ReplShell::new()),
        };

        Self::new()
            .with_arc_provider(shell_provider)
            .with_provider(super::ReadFile::new())
            .with_provider(super::ApplyPatchTool)
            .with_provider(super::Glob)
            .with_provider(super::Grep)
            .with_provider(super::Ls)
    }

    /// Default tools for a concrete execution mode.
    pub fn defaults_for(mode: ExecutionMode, deps: ToolSetDeps) -> Self {
        let mut set = Self::core_for(mode);

        if let Some(prompt_bridge) = deps.prompt_bridge.clone() {
            set = set.with_provider(super::AskTool::new(prompt_bridge));
        }

        #[cfg(feature = "sqlite-store")]
        if deps.store.is_some() {
            set = set.with_provider(super::StateStore::new(
                deps.skill_dirs.clone().unwrap_or_default(),
            ));
        }

        if let Some(key) = deps.tavily_api_key {
            set = set
                .with_provider(super::WebSearch::new(key.clone()))
                .with_provider(super::FetchUrl::new(key));
        }

        if let Some(dirs) = deps.skill_dirs {
            set = set.with_provider(super::SkillStore::new(dirs));
        }

        set
    }

    pub fn standard_defaults(deps: ToolSetDeps) -> Self {
        Self::defaults_for(ExecutionMode::Standard, deps)
    }

    pub fn repl_defaults(deps: ToolSetDeps) -> Self {
        Self::defaults_for(ExecutionMode::Repl, deps)
    }

    pub fn with_provider(mut self, provider: impl ToolProvider) -> Self {
        let arc: Arc<dyn ToolProvider> = Arc::new(provider);
        for def in arc.definitions() {
            self.tools.insert(def.name.clone(), (def, Arc::clone(&arc)));
        }
        self
    }

    pub fn with_arc_provider(mut self, provider: Arc<dyn ToolProvider>) -> Self {
        for def in provider.definitions() {
            self.tools
                .insert(def.name.clone(), (def, Arc::clone(&provider)));
        }
        self
    }

    pub fn with_toolset(mut self, other: Self) -> Self {
        for (name, entry) in other.tools {
            self.tools.insert(name, entry);
        }
        self
    }

    pub fn without_tool(mut self, tool_name: &str) -> Self {
        self.tools.remove(tool_name);
        self
    }

    pub fn without_capability(mut self, capability: CapabilityId) -> Self {
        for name in tools_for_capability(capability) {
            self.tools.remove(*name);
        }
        self
    }

    fn prompt_guides_for(&self, context: &ToolPromptContext) -> Vec<String> {
        let mut seen_providers = HashSet::new();
        let mut seen_guides = HashSet::new();
        let mut guides = Vec::new();

        for (_, provider) in self.tools.values() {
            let key = Arc::as_ptr(provider) as *const ();
            if !seen_providers.insert(key) {
                continue;
            }
            for guide in provider
                .prompt_guides(context)
                .into_iter()
                .map(|guide| guide.trim().to_string())
                .filter(|guide| !guide.is_empty())
            {
                if seen_guides.insert(guide.clone()) {
                    guides.push(guide);
                }
            }
        }

        guides
    }
}

impl Default for ToolSet {
    fn default() -> Self {
        Self::new()
    }
}

// ── ToolProvider impl ──

#[async_trait::async_trait]
impl ToolProvider for ToolSet {
    fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|(def, _)| def.clone()).collect()
    }

    fn prompt_guides(&self, context: &ToolPromptContext) -> Vec<String> {
        self.prompt_guides_for(context)
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        match self.tools.get(name) {
            Some((_, provider)) => provider.execute(name, args).await,
            None => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }

    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        match self.tools.get(name) {
            Some((_, provider)) => provider.execute_streaming(name, args, progress).await,
            None => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolParam;
    struct MockAlpha;

    #[async_trait::async_trait]
    impl ToolProvider for MockAlpha {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition {
                name: "alpha".into(),
                description: vec![crate::ToolText::new(
                    "Alpha tool",
                    [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
                )],
                params: vec![],
                returns: "str".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: true,
            }]
        }
        async fn execute(&self, _name: &str, _args: &serde_json::Value) -> ToolResult {
            ToolResult::ok(serde_json::json!("alpha_result"))
        }
    }

    struct MockBeta;

    #[async_trait::async_trait]
    impl ToolProvider for MockBeta {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition {
                name: "beta".into(),
                description: vec![crate::ToolText::new(
                    "Beta tool",
                    [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
                )],
                params: vec![ToolParam::typed("x", "int")],
                returns: "int".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: true,
            }]
        }
        async fn execute(&self, _name: &str, _args: &serde_json::Value) -> ToolResult {
            ToolResult::ok(serde_json::json!(42))
        }
    }

    /// Provider that exposes two tools under one struct.
    struct MockMulti;

    #[async_trait::async_trait]
    impl ToolProvider for MockMulti {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![
                ToolDefinition {
                    name: "multi_a".into(),
                    description: vec![],
                    params: vec![],
                    returns: "str".into(),
                    examples: vec![],
                    hidden: false,
                    inject_into_prompt: true,
                },
                ToolDefinition {
                    name: "multi_b".into(),
                    description: vec![],
                    params: vec![],
                    returns: "str".into(),
                    examples: vec![],
                    hidden: false,
                    inject_into_prompt: true,
                },
            ]
        }
        async fn execute(&self, name: &str, _args: &serde_json::Value) -> ToolResult {
            ToolResult::ok(serde_json::json!(format!("{name}_result")))
        }
    }

    struct MockGuides;

    #[async_trait::async_trait]
    impl ToolProvider for MockGuides {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![
                ToolDefinition {
                    name: "guide_a".into(),
                    description: vec![],
                    params: vec![],
                    returns: "str".into(),
                    examples: vec![],
                    hidden: false,
                    inject_into_prompt: true,
                },
                ToolDefinition {
                    name: "guide_b".into(),
                    description: vec![],
                    params: vec![],
                    returns: "str".into(),
                    examples: vec![],
                    hidden: false,
                    inject_into_prompt: true,
                },
            ]
        }

        fn prompt_guides(&self, context: &ToolPromptContext) -> Vec<String> {
            let mut guides = vec!["### Shared\nOnly once.".to_string()];
            if context.omitted_tool_count > 0 {
                guides.push("### Discovery\nOnly when omitted.".to_string());
            }
            guides
        }

        async fn execute(&self, _name: &str, _args: &serde_json::Value) -> ToolResult {
            ToolResult::ok(serde_json::json!("ok"))
        }
    }

    #[test]
    fn mode_specific_core_toolsets_expose_different_shell_surfaces() {
        let standard = ToolSet::core_for(crate::ExecutionMode::Standard);
        let repl = ToolSet::core_for(crate::ExecutionMode::Repl);

        let standard_defs = standard
            .definitions()
            .into_iter()
            .map(|def| def.name)
            .collect::<std::collections::BTreeSet<_>>();
        let repl_defs = repl
            .definitions()
            .into_iter()
            .map(|def| def.name)
            .collect::<std::collections::BTreeSet<_>>();

        assert!(standard_defs.contains("exec_command"));
        assert!(standard_defs.contains("write_stdin"));
        assert!(!standard_defs.contains("shell"));

        assert!(repl_defs.contains("shell"));
        assert!(repl_defs.contains("shell_wait"));
        assert!(!repl_defs.contains("exec_command"));
    }

    #[test]
    fn defaults_include_ask_when_prompt_bridge_is_available() {
        let prompt_bridge = crate::PromptBridge::new();
        let interactive = ToolSet::standard_defaults(ToolSetDeps {
            prompt_bridge: Some(prompt_bridge.clone()),
            ..Default::default()
        });
        let no_prompt = ToolSet::standard_defaults(ToolSetDeps {
            prompt_bridge: None,
            ..Default::default()
        });
        let repl = ToolSet::repl_defaults(ToolSetDeps {
            prompt_bridge: Some(prompt_bridge),
            ..Default::default()
        });
        let repl_no_prompt = ToolSet::repl_defaults(ToolSetDeps {
            prompt_bridge: None,
            ..Default::default()
        });

        assert!(interactive.definitions().iter().any(|d| d.name == "ask"));
        assert!(!no_prompt.definitions().iter().any(|d| d.name == "ask"));
        assert!(repl.definitions().iter().any(|d| d.name == "ask"));
        assert!(!repl_no_prompt.definitions().iter().any(|d| d.name == "ask"));
    }

    #[test]
    fn standard_defaults_do_not_expose_batch_tool() {
        let standard = ToolSet::standard_defaults(ToolSetDeps::default());
        assert!(!standard.definitions().iter().any(|d| d.name == "batch"));
    }

    #[test]
    fn new_is_empty() {
        let set = ToolSet::new();
        assert!(set.definitions().is_empty());
    }

    #[test]
    fn add_provider_via_method() {
        let set = ToolSet::new().with_provider(MockAlpha);
        let defs = set.definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "alpha");
    }

    #[test]
    fn add_arc_provider() {
        let arc: Arc<dyn ToolProvider> = Arc::new(MockBeta);
        let set = ToolSet::new().with_arc_provider(arc);
        let defs = set.definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "beta");
    }

    #[test]
    fn sub_str_removes_tool() {
        let set = ToolSet::new()
            .with_provider(MockAlpha)
            .with_provider(MockBeta)
            .without_tool("alpha");
        let names: Vec<String> = set.definitions().iter().map(|d| d.name.clone()).collect();
        assert_eq!(names, vec!["beta"]);
    }

    #[test]
    fn sub_str_missing_is_noop() {
        let set = ToolSet::new()
            .with_provider(MockAlpha)
            .without_tool("nonexistent");
        assert_eq!(set.definitions().len(), 1);
    }

    #[test]
    fn sub_capability_removes_group() {
        // Shell capability tools: shell, shell_wait, shell_read, shell_write, shell_kill
        // MockAlpha's "alpha" tool is NOT in the Shell capability, so it survives.
        let set = ToolSet::new()
            .with_provider(MockAlpha)
            .without_capability(CapabilityId::Shell);
        let names: Vec<String> = set.definitions().iter().map(|d| d.name.clone()).collect();
        assert_eq!(names, vec!["alpha"]);
    }

    #[test]
    fn method_chaining() {
        let set = ToolSet::new()
            .with_provider(MockAlpha)
            .with_provider(MockBeta)
            .without_tool("alpha")
            .with_provider(MockMulti)
            .without_tool("multi_b");
        let mut names: Vec<String> = set.definitions().iter().map(|d| d.name.clone()).collect();
        names.sort();
        assert_eq!(names, vec!["beta", "multi_a"]);
    }

    #[test]
    fn duplicate_name_last_wins() {
        // MockAlpha defines "alpha". Adding another provider that also defines "alpha"
        // should replace the first.
        struct MockAlpha2;

        #[async_trait::async_trait]
        impl ToolProvider for MockAlpha2 {
            fn definitions(&self) -> Vec<ToolDefinition> {
                vec![ToolDefinition {
                    name: "alpha".into(),
                    description: vec![crate::ToolText::new(
                        "Alpha v2",
                        [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
                    )],
                    params: vec![],
                    returns: "str".into(),
                    examples: vec![],
                    hidden: false,
                    inject_into_prompt: true,
                }]
            }
            async fn execute(&self, _name: &str, _args: &serde_json::Value) -> ToolResult {
                ToolResult::ok(serde_json::json!("alpha_v2"))
            }
        }

        let set = ToolSet::new()
            .with_provider(MockAlpha)
            .with_provider(MockAlpha2);
        let defs = set.definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(
            defs[0].description_for(crate::ExecutionMode::Repl),
            "Alpha v2"
        );
    }

    #[test]
    fn multi_tool_provider_shares_arc() {
        let set = ToolSet::new().with_provider(MockMulti);
        let defs = set.definitions();
        assert_eq!(defs.len(), 2);
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"multi_a"));
        assert!(names.contains(&"multi_b"));
    }

    #[test]
    fn prompt_guides_are_deduped_per_provider_and_content() {
        let set = ToolSet::new().with_provider(MockGuides);
        let guides = set.prompt_guides(&ToolPromptContext {
            mode: crate::ExecutionMode::Repl,
            omitted_tool_count: 1,
        });
        assert_eq!(
            guides,
            vec![
                "### Shared\nOnly once.".to_string(),
                "### Discovery\nOnly when omitted.".to_string()
            ]
        );
    }

    #[test]
    fn prompt_guides_receive_context() {
        let set = ToolSet::new().with_provider(MockGuides);
        let guides = set.prompt_guides(&ToolPromptContext {
            mode: crate::ExecutionMode::Repl,
            omitted_tool_count: 0,
        });
        assert_eq!(guides, vec!["### Shared\nOnly once.".to_string()]);
    }

    #[test]
    fn toolset_combines_toolsets() {
        let a = ToolSet::new().with_provider(MockAlpha);
        let b = ToolSet::new().with_provider(MockBeta);
        let combined = a.with_toolset(b);
        let mut names: Vec<String> = combined
            .definitions()
            .iter()
            .map(|d| d.name.clone())
            .collect();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[tokio::test]
    async fn execute_routes_correctly() {
        let set = ToolSet::new()
            .with_provider(MockAlpha)
            .with_provider(MockBeta);
        let r = set.execute("alpha", &serde_json::json!({})).await;
        assert!(r.success);
        assert_eq!(r.result, serde_json::json!("alpha_result"));

        let r = set.execute("beta", &serde_json::json!({"x": 1})).await;
        assert!(r.success);
        assert_eq!(r.result, serde_json::json!(42));
    }

    #[tokio::test]
    async fn execute_unknown_tool_errors() {
        let set = ToolSet::new().with_provider(MockAlpha);
        let r = set.execute("nonexistent", &serde_json::json!({})).await;
        assert!(!r.success);
    }

    #[tokio::test]
    async fn execute_multi_tool_provider() {
        let set = ToolSet::new().with_provider(MockMulti);
        let r = set.execute("multi_a", &serde_json::json!({})).await;
        assert!(r.success);
        assert_eq!(r.result, serde_json::json!("multi_a_result"));

        let r = set.execute("multi_b", &serde_json::json!({})).await;
        assert!(r.success);
        assert_eq!(r.result, serde_json::json!("multi_b_result"));
    }

    #[tokio::test]
    async fn removed_tool_is_not_executable() {
        let set = ToolSet::new()
            .with_provider(MockAlpha)
            .with_provider(MockBeta)
            .without_tool("alpha");
        let r = set.execute("alpha", &serde_json::json!({})).await;
        assert!(!r.success);
    }
}
