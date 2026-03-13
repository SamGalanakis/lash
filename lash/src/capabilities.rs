use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::LazyLock;

use crate::ToolDefinition;

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityId {
    CoreRead,
    CoreWrite,
    Shell,
    Planning,
    Delegation,
    Memory,
    History,
    Skills,
    Web,
}

impl CapabilityId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CoreRead => "core_read",
            Self::CoreWrite => "core_write",
            Self::Shell => "shell",
            Self::Planning => "planning",
            Self::Delegation => "delegation",
            Self::Memory => "memory",
            Self::History => "history",
            Self::Skills => "skills",
            Self::Web => "web",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "core_read" => Some(Self::CoreRead),
            "core_write" => Some(Self::CoreWrite),
            "shell" => Some(Self::Shell),
            "planning" => Some(Self::Planning),
            "delegation" => Some(Self::Delegation),
            "memory" => Some(Self::Memory),
            "history" => Some(Self::History),
            "skills" => Some(Self::Skills),
            "web" => Some(Self::Web),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AgentCapabilities {
    #[serde(default)]
    pub enabled_capabilities: BTreeSet<CapabilityId>,
    #[serde(default)]
    pub enabled_tools: BTreeSet<String>,
}

impl AgentCapabilities {
    pub fn enabled(&self, capability: CapabilityId) -> bool {
        self.enabled_capabilities.contains(&capability)
    }

    pub fn enable(mut self, capability: CapabilityId) -> Self {
        self.enabled_capabilities.insert(capability);
        self
    }

    pub fn disable(mut self, capability: CapabilityId) -> Self {
        self.enabled_capabilities.remove(&capability);
        self
    }

    pub fn with_tool(mut self, name: impl Into<String>) -> Self {
        self.enabled_tools.insert(name.into());
        self
    }
}

impl Default for AgentCapabilities {
    fn default() -> Self {
        Self {
            enabled_capabilities: default_enabled_capabilities(),
            enabled_tools: BTreeSet::new(),
        }
    }
}

pub struct CapabilityDefinition {
    pub id: CapabilityId,
    pub name: &'static str,
    pub description: &'static str,
    pub prompt_section: Option<&'static str>,
    pub helper_bindings: &'static [&'static str],
    pub tools: &'static [&'static str],
    pub enabled_by_default: bool,
}

pub const CAPABILITY_DEFINITIONS: &[CapabilityDefinition] = &[
    CapabilityDefinition {
        id: CapabilityId::CoreRead,
        name: "Core Read",
        description: "Core file navigation and discovery.",
        prompt_section: Some(
            "### Orient -> Read -> Act\n1. Use `ls` / `glob` to orient yourself first\n2. Use `read_file` / `grep` for content-level context before mutating files\n3. Apply focused changes only after you understand the surrounding code",
        ),
        helper_bindings: &[],
        tools: &["read_file", "glob", "grep", "ls", "search_tools"],
        enabled_by_default: true,
    },
    CapabilityDefinition {
        id: CapabilityId::CoreWrite,
        name: "Core Write",
        description: "Primary file mutation tool.",
        prompt_section: None,
        helper_bindings: &[],
        tools: &["apply_patch"],
        enabled_by_default: true,
    },
    CapabilityDefinition {
        id: CapabilityId::Shell,
        name: "Shell",
        description: "Command execution and interactive process handles.",
        prompt_section: Some(
            "### Git Safety\nDo not revert user changes you did not make. Avoid destructive git commands unless explicitly requested.",
        ),
        helper_bindings: &[],
        tools: &[
            "exec_command",
            "write_stdin",
            "shell",
            "shell_wait",
            "shell_read",
            "shell_write",
            "shell_kill",
        ],
        enabled_by_default: true,
    },
    CapabilityDefinition {
        id: CapabilityId::Planning,
        name: "Planning",
        description: "Native plan tracking for substantial multi-step work.",
        prompt_section: Some(
            "### Planning\nUse `update_plan` for substantial multi-step work, not for trivial or single-step requests. Keep plans short, concrete, and easy to verify. There should be at most one `in_progress` step at a time; mark completed work promptly and keep the next active step current while you work. Do not restate the full plan after calling `update_plan`; the runtime already surfaces it. `update_plan` is a checklist tool for normal execution turns, not for plan mode.",
        ),
        helper_bindings: &[],
        tools: &["update_plan", "ask"],
        enabled_by_default: true,
    },
    CapabilityDefinition {
        id: CapabilityId::Delegation,
        name: "Delegation",
        description: "Sub-agent orchestration via agent_call.",
        prompt_section: Some(
            "### Delegation\nUse `agent_call` for scoped sub-tasks. Prefer low-intelligence delegates for read-only lookup/summarization work, and avoid overlapping file edits across concurrent delegates.",
        ),
        helper_bindings: &[],
        tools: &["agent_call", "agent_result", "agent_kill"],
        enabled_by_default: true,
    },
    CapabilityDefinition {
        id: CapabilityId::Skills,
        name: "Skills",
        description: "Skill discovery and loading.",
        prompt_section: Some(
            "### Skills\nWhen the user requests a skill-based workflow, use `search_skills(...)` to discover the right skill, then `load_skill(name)` and follow the skill instructions.",
        ),
        helper_bindings: &["search_skills"],
        tools: &["load_skill", "read_skill_file", "search_skills"],
        enabled_by_default: true,
    },
    CapabilityDefinition {
        id: CapabilityId::Web,
        name: "Web",
        description: "Web search and page fetch tools.",
        prompt_section: None,
        helper_bindings: &[],
        tools: &["search_web", "fetch_url"],
        enabled_by_default: true,
    },
];

pub fn default_enabled_capabilities() -> BTreeSet<CapabilityId> {
    let enabled = CAPABILITY_DEFINITIONS
        .iter()
        .filter(|d| d.enabled_by_default)
        .map(|d| d.id)
        .collect::<BTreeSet<_>>();
    #[cfg(feature = "sqlite-store")]
    {
        let mut enabled = enabled;
        enabled.insert(CapabilityId::History);
        enabled.insert(CapabilityId::Memory);
        enabled
    }
    #[cfg(not(feature = "sqlite-store"))]
    {
        enabled
    }
}

pub fn capability_def(id: CapabilityId) -> Option<&'static CapabilityDefinition> {
    CAPABILITY_DEFINITIONS.iter().find(|d| d.id == id)
}

pub fn tools_for_capability(id: CapabilityId) -> &'static [&'static str] {
    capability_def(id).map(|d| d.tools).unwrap_or(&[])
}

pub fn helper_bindings_for_capability(id: CapabilityId) -> &'static [&'static str] {
    capability_def(id).map(|d| d.helper_bindings).unwrap_or(&[])
}

pub fn prompt_sections_for_capabilities(
    enabled: &BTreeSet<CapabilityId>,
    helper_bindings: &BTreeSet<String>,
    available_tools: &BTreeSet<String>,
) -> Vec<String> {
    let mut sections = Vec::new();

    for id in enabled {
        let Some(def) = capability_def(*id) else {
            continue;
        };
        let has_surface = def.tools.iter().any(|tool| available_tools.contains(*tool))
            || def
                .helper_bindings
                .iter()
                .any(|binding| helper_bindings.contains(*binding));
        if has_surface
            && let Some(section) = def.prompt_section
            && !section.trim().is_empty()
        {
            sections.push(section.to_string());
        }
    }

    if enabled.contains(&CapabilityId::Delegation) && available_tools.contains("agent_call") {
        let recall_line = if helper_bindings.contains("search_history")
            && helper_bindings.contains("search_mem")
        {
            "When prior context likely matters, use a low-intelligence (read-only) `agent_call` to summarize relevant `search_history` or memory results quickly, then continue execution."
        } else if helper_bindings.contains("search_history") {
            "When prior context likely matters, use a low-intelligence (read-only) `agent_call` to summarize relevant `search_history` results quickly, then continue execution."
        } else if helper_bindings.contains("search_mem") {
            "When prior context likely matters, use a low-intelligence (read-only) `agent_call` to summarize relevant memory results quickly, then continue execution."
        } else {
            "When prior context likely matters, use a low-intelligence (read-only) `agent_call` to summarize relevant context quickly, then continue execution."
        };
        sections.push(format!(
            "### Delegation Recall\n{}\nDo not delegate straightforward local file or image inspection that you can do directly with available tools.",
            recall_line
        ));
    }

    sections
}

pub fn capability_for_tool(name: &str) -> Option<CapabilityId> {
    static TOOL_TO_CAP: LazyLock<HashMap<&'static str, CapabilityId>> = LazyLock::new(|| {
        let mut out = HashMap::new();
        for def in CAPABILITY_DEFINITIONS {
            for tool in def.tools {
                out.insert(*tool, def.id);
            }
        }
        out
    });
    TOOL_TO_CAP.get(name).copied()
}

pub struct ResolvedFeatures {
    pub enabled_capabilities: BTreeSet<CapabilityId>,
    pub effective_tools: BTreeSet<String>,
    pub helper_bindings: BTreeSet<String>,
}

pub fn resolve_features(
    caps: &AgentCapabilities,
    available_defs: &[ToolDefinition],
) -> ResolvedFeatures {
    let available: HashSet<&str> = available_defs.iter().map(|d| d.name.as_str()).collect();
    let mut effective_tools = BTreeSet::new();
    let mut helper_bindings = BTreeSet::new();
    for id in &caps.enabled_capabilities {
        for tool in tools_for_capability(*id) {
            if available.contains(*tool) {
                effective_tools.insert((*tool).to_string());
            }
        }
        for binding in helper_bindings_for_capability(*id) {
            if available.contains(*binding) {
                helper_bindings.insert((*binding).to_string());
            }
        }
    }
    for tool in &caps.enabled_tools {
        if available.contains(tool.as_str()) {
            effective_tools.insert(tool.clone());
        }
    }
    ResolvedFeatures {
        enabled_capabilities: caps.enabled_capabilities.clone(),
        effective_tools,
        helper_bindings,
    }
}

#[cfg(test)]
mod prompt_tests {
    use super::*;

    #[test]
    fn capability_prompt_sections_only_include_available_surfaces() {
        let enabled = BTreeSet::from([CapabilityId::Planning]);
        let sections =
            prompt_sections_for_capabilities(&enabled, &BTreeSet::new(), &BTreeSet::new());
        assert!(sections.is_empty());

        let available_tools = BTreeSet::from(["update_plan".to_string()]);
        let sections =
            prompt_sections_for_capabilities(&enabled, &BTreeSet::new(), &available_tools);
        assert_eq!(sections.len(), 1);
        assert!(sections[0].contains("### Planning"));
    }

    #[test]
    fn delegation_prompt_sections_include_recall_guidance_when_helpers_exist() {
        let enabled = BTreeSet::from([CapabilityId::Delegation]);
        let helpers = BTreeSet::from(["search_history".to_string(), "search_mem".to_string()]);
        let tools = BTreeSet::from(["agent_call".to_string(), "agent_result".to_string()]);
        let sections = prompt_sections_for_capabilities(&enabled, &helpers, &tools);
        assert!(
            sections
                .iter()
                .any(|section| section.contains("### Delegation"))
        );
        assert!(
            sections
                .iter()
                .any(|section| section.contains("### Delegation Recall"))
        );
        assert!(
            sections
                .iter()
                .any(|section| section.contains("`search_history` or memory results"))
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defs(names: &[&str]) -> Vec<ToolDefinition> {
        names
            .iter()
            .map(|n| ToolDefinition {
                name: (*n).to_string(),
                description: vec![crate::ToolText::new(
                    "test",
                    [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
                )],
                params: vec![],
                returns: "any".to_string(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            })
            .collect()
    }

    #[test]
    fn explicit_tool_is_included_without_capability_helpers() {
        let mut caps = AgentCapabilities::default().disable(CapabilityId::Skills);
        caps = caps.with_tool("load_skill");
        let resolved = resolve_features(&caps, &defs(&["load_skill", "search_skills"]));
        assert!(resolved.effective_tools.contains("load_skill"));
        assert!(!resolved.helper_bindings.contains("search_skills"));
    }

    #[test]
    fn plugin_capabilities_are_not_resolved_from_static_definitions() {
        let caps = AgentCapabilities::default()
            .enable(CapabilityId::History)
            .enable(CapabilityId::Memory);
        let resolved = resolve_features(
            &caps,
            &defs(&["search_history", "search_mem", "mem_set", "mem_all"]),
        );
        assert!(!resolved.effective_tools.contains("search_history"));
        assert!(!resolved.effective_tools.contains("search_mem"));
        assert!(
            resolved
                .enabled_capabilities
                .contains(&CapabilityId::History)
        );
        assert!(
            resolved
                .enabled_capabilities
                .contains(&CapabilityId::Memory)
        );
    }

    #[test]
    fn unavailable_tools_are_not_resolved() {
        let defs = vec![];
        let caps = AgentCapabilities::default();
        let resolved = resolve_features(&caps, &defs);
        assert!(!resolved.effective_tools.contains("shell"));
    }
}
