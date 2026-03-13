use std::collections::{BTreeSet, HashMap};
use std::str::FromStr;

use crate::ExecutionMode;
use crate::plugin::PromptContribution;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptSectionName {
    Identity,
    Environment,
    Personality,
    ExecutionContract,
    TerminationContract,
    ToolAccess,
    ToolGuides,
    AvailableTools,
    ErrorRecovery,
    Builtins,
    Memory,
    MemoryApi,
    PluginExtensions,
    ProjectInstructions,
    Guidelines,
}

impl PromptSectionName {
    pub const ALL: [Self; 15] = [
        Self::Identity,
        Self::Personality,
        Self::ExecutionContract,
        Self::TerminationContract,
        Self::ErrorRecovery,
        Self::Guidelines,
        Self::ProjectInstructions,
        Self::Environment,
        Self::Memory,
        Self::MemoryApi,
        Self::ToolAccess,
        Self::Builtins,
        Self::ToolGuides,
        Self::PluginExtensions,
        Self::AvailableTools,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Environment => "environment",
            Self::Personality => "personality",
            Self::ExecutionContract => "execution_contract",
            Self::TerminationContract => "termination_contract",
            Self::ToolAccess => "tool_access",
            Self::ToolGuides => "tool_guides",
            Self::AvailableTools => "available_tools",
            Self::ErrorRecovery => "error_recovery",
            Self::Builtins => "builtins",
            Self::Memory => "memory",
            Self::MemoryApi => "memory_api",
            Self::PluginExtensions => "plugin_extensions",
            Self::ProjectInstructions => "project_instructions",
            Self::Guidelines => "guidelines",
        }
    }

    pub fn names_csv() -> String {
        Self::ALL
            .iter()
            .map(Self::as_str)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl FromStr for PromptSectionName {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let value = raw.trim().to_ascii_lowercase();
        match value.as_str() {
            "identity" => Ok(Self::Identity),
            "environment" => Ok(Self::Environment),
            "personality" => Ok(Self::Personality),
            "execution_contract" => Ok(Self::ExecutionContract),
            "termination_contract" => Ok(Self::TerminationContract),
            "tool_access" => Ok(Self::ToolAccess),
            "tool_guides" => Ok(Self::ToolGuides),
            "available_tools" => Ok(Self::AvailableTools),
            "error_recovery" => Ok(Self::ErrorRecovery),
            "builtins" => Ok(Self::Builtins),
            "memory" => Ok(Self::Memory),
            "memory_api" => Ok(Self::MemoryApi),
            "plugin_extensions" => Ok(Self::PluginExtensions),
            "project_instructions" => Ok(Self::ProjectInstructions),
            "guidelines" => Ok(Self::Guidelines),
            _ => Err(format!(
                "unknown prompt section `{raw}` (expected one of: {})",
                Self::names_csv()
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptOverrideMode {
    Replace,
    Prepend,
    Append,
    Disable,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PromptSectionOverride {
    pub section: PromptSectionName,
    pub mode: PromptOverrideMode,
    #[serde(default)]
    pub content: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptProfile {
    RootInteractive,
    RootHeadless,
    SubAgentInteractive,
    SubAgentHeadless,
}

impl PromptProfile {
    pub fn from_flags(headless: bool, sub_agent: bool) -> Self {
        match (headless, sub_agent) {
            (false, false) => Self::RootInteractive,
            (true, false) => Self::RootHeadless,
            (false, true) => Self::SubAgentInteractive,
            (true, true) => Self::SubAgentHeadless,
        }
    }

    pub fn is_headless(self) -> bool {
        matches!(self, Self::RootHeadless | Self::SubAgentHeadless)
    }

    pub fn is_subagent(self) -> bool {
        matches!(self, Self::SubAgentInteractive | Self::SubAgentHeadless)
    }
}

pub struct PromptComposeInput<'a> {
    pub profile: PromptProfile,
    pub execution_mode: ExecutionMode,
    pub context: &'a str,
    pub tool_list: &'a str,
    pub tool_names: &'a [String],
    pub helper_bindings: &'a BTreeSet<String>,
    pub guide_sections: &'a [String],
    pub plugin_prompt_contributions: &'a [PromptContribution],
    pub can_write: bool,
    pub include_soul: bool,
    pub project_instructions: &'a str,
    pub overrides: &'a [PromptSectionOverride],
}

fn has_helper(input: &PromptComposeInput<'_>, helper: &str) -> bool {
    input.helper_bindings.contains(helper)
}

fn history_enabled(input: &PromptComposeInput<'_>) -> bool {
    has_helper(input, "search_history")
        || input.tool_names.iter().any(|name| name == "search_history")
}

fn memory_enabled(input: &PromptComposeInput<'_>) -> bool {
    has_helper(input, "search_mem")
        || input.tool_names.iter().any(|name| {
            matches!(
                name.as_str(),
                "search_mem" | "mem_set" | "mem_get" | "mem_delete" | "mem_all"
            )
        })
}

pub fn compose_system_prompt(input: PromptComposeInput<'_>) -> String {
    let mut sections: HashMap<PromptSectionName, Option<String>> = HashMap::new();
    for section in PromptSectionName::ALL {
        sections.insert(section, default_section(section, &input));
    }
    apply_prompt_contributions(&mut sections, input.plugin_prompt_contributions);
    apply_overrides(&mut sections, input.overrides);

    PromptSectionName::ALL
        .iter()
        .filter_map(|section| sections.get(section).cloned().flatten())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn apply_overrides(
    sections: &mut HashMap<PromptSectionName, Option<String>>,
    overrides: &[PromptSectionOverride],
) {
    for ov in overrides {
        let entry = sections.entry(ov.section).or_insert(None);
        match ov.mode {
            PromptOverrideMode::Disable => {
                *entry = None;
            }
            PromptOverrideMode::Replace => {
                *entry = Some(ov.content.clone());
            }
            PromptOverrideMode::Prepend => {
                let existing = entry.take().unwrap_or_default();
                *entry = Some(if existing.trim().is_empty() {
                    ov.content.clone()
                } else {
                    format!("{}\n\n{}", ov.content, existing)
                });
            }
            PromptOverrideMode::Append => {
                let existing = entry.take().unwrap_or_default();
                *entry = Some(if existing.trim().is_empty() {
                    ov.content.clone()
                } else {
                    format!("{}\n\n{}", existing, ov.content)
                });
            }
        }
    }
}

fn apply_prompt_contributions(
    sections: &mut HashMap<PromptSectionName, Option<String>>,
    contributions: &[PromptContribution],
) {
    for contribution in contributions {
        let content = contribution.content.trim();
        if content.is_empty() {
            continue;
        }
        let entry = sections.entry(contribution.section).or_insert(None);
        match entry {
            Some(existing) if existing.trim().is_empty() => {
                *existing = content.to_string();
            }
            Some(existing) => {
                existing.push_str("\n\n");
                existing.push_str(content);
            }
            None => {
                *entry = Some(content.to_string());
            }
        }
    }
}

fn default_section(section: PromptSectionName, input: &PromptComposeInput<'_>) -> Option<String> {
    let profile = input.profile;
    match section {
        PromptSectionName::Identity => Some(match profile {
            PromptProfile::RootInteractive => match input.execution_mode {
                ExecutionMode::Repl => "You are an AI coding assistant operating in a persistent REPL with tool access.\nYou power lash, a terminal-based coding agent. Understand the codebase, make changes, run commands, and report outcomes clearly.".to_string(),
                ExecutionMode::Standard => "You are an AI coding assistant with direct tool-calling access.\nYou power lash, a terminal-based coding agent. Understand the codebase, make changes, run commands, and report outcomes clearly.".to_string(),
            },
            PromptProfile::RootHeadless => "You are an autonomous AI coding agent running in non-interactive mode.\nComplete the task end-to-end without asking for user input.".to_string(),
            PromptProfile::SubAgentInteractive => match input.execution_mode {
                ExecutionMode::Repl => {
                    if input.can_write {
                        "You are a sub-agent inside lash working on a delegated task.\nUse tools decisively and return results to the caller via `finish ...` when complete.".to_string()
                    } else {
                        "You are a read-only sub-agent inside lash working on a delegated task.\nFocus on lookup/summarization tasks and return results to the caller via `finish ...` when complete.".to_string()
                    }
                }
                ExecutionMode::Standard => {
                    if input.can_write {
                        "You are a sub-agent inside lash working on a delegated task.\nUse tools decisively and return a final answer to the caller when complete.".to_string()
                    } else {
                        "You are a read-only sub-agent inside lash working on a delegated task.\nFocus on lookup/summarization tasks and return a final answer to the caller when complete.".to_string()
                    }
                }
            },
            PromptProfile::SubAgentHeadless => match input.execution_mode {
                ExecutionMode::Repl => {
                    if input.can_write {
                        "You are a headless sub-agent inside lash working on a delegated task.\nOperate autonomously and return final results via `finish ...` only when complete.".to_string()
                    } else {
                        "You are a headless read-only sub-agent inside lash working on a delegated task.\nOperate autonomously on lookup/summarization tasks and return final results via `finish ...` only when complete.".to_string()
                    }
                }
                ExecutionMode::Standard => {
                    if input.can_write {
                        "You are a headless sub-agent inside lash working on a delegated task.\nOperate autonomously and return a final answer only when complete.".to_string()
                    } else {
                        "You are a headless read-only sub-agent inside lash working on a delegated task.\nOperate autonomously on lookup/summarization tasks and return a final answer only when complete.".to_string()
                    }
                }
            },
        }),
        PromptSectionName::Environment => Some(format!("## Environment\n\n{}", input.context)),
        PromptSectionName::Personality => {
            if !input.include_soul {
                None
            } else {
                Some(
                    "## Core Principles\n\n- First-principles thinker\n- Allergic to accidental complexity\n- Direct over diplomatic\n- Skeptical of abstraction\n- Show, don't lecture\n- High standards by default"
                        .to_string(),
                )
            }
        }
        PromptSectionName::ExecutionContract => Some(format!(
            "{}\n{}",
            if matches!(input.execution_mode, ExecutionMode::Repl) {
                "## Execution Contract\n\nYour output can include prose and `<repl>` blocks.\n- Use prose only when no execution is needed\n- `<repl>` executes immediately when `</repl>` is reached\n- `finish expr` ends the turn, even when the value renders as empty text\n- `observe expr` does not end the turn; it sends hidden feedback to the next model step\n- A `<repl>` block may also do work silently and continue without `finish` or `observe`\n- Do not assume prose after `</repl>` is user-visible unless you pass it via `finish ...`\n- Maximum one `<repl>` block per response\n- For direct conversational requests that need no tools, respond in prose only\n- The REPL has no `print`; use `observe expr` for intermediate inspection and `finish ...` for the final answer\n- Prefer small REPL blocks over giant one-shot programs\n- Validate one or two steps, inspect intermediate outputs, then extend the workflow\n- Only `observe expr` creates intermediate execution feedback for the next model step; ordinary assignments are silent\n- The user should see only final prose or `finish ...` output, not intermediate observations\n\n### REPL Language\n\nThe REPL is `lashlang`, a small workflow language for tool orchestration.\n- Values are null, booleans, numbers, strings, lists, and records\n- Assign with `name = expr`\n- Call tools with `call tool_name { arg: expr }`\n- Use `observe expr` to inspect a value and continue execution\n- Use `finish expr` to end the turn and return a final answer\n- Control flow is limited to statement `if`, `for`, `parallel`, and `finish`\n- Use ternary expressions for inline branching: `cond ? yes : no`\n- Boolean negation supports both `!cond` and `not cond`\n- Boolean conjunction/disjunction support both `&&` / `||` and `and` / `or`\n- Tool results are records like `{ ok: true, value: ... }` or `{ ok: false, error: ... }`\n- Access the wrapped payload via `.value` only when `result.ok` is true\n- Do not assume every `value` is a record: many tools return strings, numbers, or lists directly\n- There are no imports, classes, methods, exceptions, comprehensions, or arbitrary standard library access\n- Use builtins like `len(...)`, `empty(...)`, `contains(...)`, `slice(...)`, `json_parse(...)`, and `format(...)`\n- `format(value)` stringifies a single value; `format(\"...\", args...)` formats with placeholders\n- String `+` concatenation auto-stringifies when either side is already a string\n- `format(...)` already stringifies records/lists/errors for display; do not wrap values in `json_stringify(...)`\n- If you need unsupported features, use the appropriate host tool instead of emulating them inside the REPL"
            } else {
                "## Execution Contract\n\nUse direct tool calls when execution is needed.\n- Do not emit `<repl>` blocks or Python code\n- Call tools directly with valid arguments\n- Group independent tool calls in the same response; serialize only when later arguments depend on earlier results\n- Avoid filler prose between tool calls\n- Keep going until the task is complete; do not stop after inspection or partial progress\n- If you are unsure, inspect or validate more instead of guessing\n- For direct conversational requests that need no tools, respond in prose only"
            },
            if matches!(input.execution_mode, ExecutionMode::Repl) {
                if profile.is_headless() {
                    "- In headless mode, prose-only turns are invalid; execute via `<repl>`"
                } else {
                    "- In interactive mode, prose-only is fine only if you never opened `<repl>`; after REPL execution, use `observe ...` to continue or `finish ...` when your final answer is ready"
                }
            } else if profile.is_headless() {
                "- In headless mode, keep calling tools until the task is complete, then return a final answer without extra commentary."
            } else {
                "- In interactive mode, return a final user-facing answer once the task is complete."
            }
        )),
        PromptSectionName::TerminationContract => Some(format!(
            "{}\n{}",
            if matches!(input.execution_mode, ExecutionMode::Repl) {
                "## Termination Contract\n\n`finish expr` ends the turn and returns control.\n- `finish ...` may only be used inside `<repl>`\n- If you never used `<repl>` in the turn, you may finish with plain prose instead\n- If you used `<repl>` at any point in the turn, only `finish ...` ends the turn\n- Do not use `finish ...` for status updates"
            } else {
                "## Termination Contract\n\nFinish by returning a final assistant answer.\n- Do not emit fake tool calls or placeholder arguments\n- Do not stop after tool execution unless the task is actually complete"
            },
            if matches!(input.execution_mode, ExecutionMode::Repl) {
                if profile.is_headless() {
                    "- Headless: call `finish ...` only after the task is fully completed"
                } else {
                    "- Interactive: `finish ...` ends the turn; otherwise the turn may continue after `observe ...` or silent REPL work"
                }
            } else if profile.is_headless() {
                "- Headless: do not stop on prose-only intermediate steps."
            } else {
                "- Interactive: provide a concise final answer when complete."
            }
        )),
        PromptSectionName::ToolAccess => Some(
            if matches!(input.execution_mode, ExecutionMode::Repl) {
                "## Tool Access\n\n- Call tools as `call tool_name { arg: expr }`\n- Tool results are wrapped records; read `result.ok`, `result.value`, and `result.error`\n- There is no `T` namespace, no imports, no methods, and no wrapper classes; work with plain records/lists/primitives\n- Use `parallel { ... }` for independent tool calls"
                    .to_string()
            } else {
                "## Tool Access\n\n- The runtime exposes only the listed tools\n- Use only tools shown in Available Tools\n- Group independent calls in one response when the provider supports it\n- Good fits: reading several files, multiple searches, and unrelated diagnostics\n- Do not parallelize dependent steps or ordered stateful work\n- Never invent tool names or arguments".to_string()
            },
        ),
        PromptSectionName::ToolGuides => {
            let guides = render_guide_sections(input.guide_sections);
            if guides.is_empty() {
                None
            } else {
                Some(format!("## Tool Guide\n\n{}", guides))
            }
        }
        PromptSectionName::AvailableTools => Some(format!("## Available Tools\n\n{}", input.tool_list)),
        PromptSectionName::ErrorRecovery => Some(
            if matches!(input.execution_mode, ExecutionMode::Repl) {
                "## Error Recovery\n\nTool failures return error records from `call`.\n- Inspect `result.ok` / `result.error` and fix the cause before retrying\n- Do not repeat failing calls unchanged\n- If a REPL program fails, shrink the block, use `observe expr` on the next uncertain value, and rerun a smaller `<repl>` block before expanding again\n- When a workflow is uncertain, validate the next small step instead of writing the whole program at once"
                    .to_string()
            } else {
                "## Error Recovery\n\nTool failures return structured errors.\n- Read the error carefully and fix the cause before retrying\n- Do not repeat failing calls unchanged\n- If a tool result is incomplete or ambiguous, inspect with other tools instead of guessing"
                    .to_string()
            },
        ),
        PromptSectionName::Builtins => {
            if matches!(input.execution_mode, ExecutionMode::Repl) {
                Some(builtins_section(
                    profile,
                    input.tool_names,
                    history_enabled(input),
                    memory_enabled(input),
                ))
            } else {
                None
            }
        }
        PromptSectionName::Memory => {
            let history_enabled = history_enabled(input);
            let memory_enabled = memory_enabled(input);
            if history_enabled || memory_enabled {
                Some(memory_section(
                    profile.is_subagent(),
                    history_enabled,
                    memory_enabled,
                ))
            } else {
                None
            }
        }
        PromptSectionName::MemoryApi => {
            if memory_enabled(input) {
                Some(memory_api_section())
            } else {
                None
            }
        }
        PromptSectionName::ProjectInstructions => {
            if input.project_instructions.trim().is_empty() {
                None
            } else {
                Some(format!(
                    "## Project Instructions\n\n{}",
                    input.project_instructions
                ))
            }
        }
        PromptSectionName::PluginExtensions => None,
        PromptSectionName::Guidelines => Some(format!(
            "## Guidelines\n\n- Bias toward concrete execution over planning chatter\n- Keep going until the request is resolved; do not stop at reconnaissance when a concrete deliverable is requested\n- Validate the smallest relevant thing first, then broaden if needed\n- Do not fix unrelated failures uncovered during validation; report them instead\n- For substantial scripts/workflows, create files and run them with host tooling\n- Use isolated environments only when required dependencies are missing\n- Avoid redundant file reads when values already exist in variables\n- Never speculate about files you have not read\n- Be concise and action-oriented\n{}\n{}",
            if !input.can_write {
                "- This agent is read-only: do not modify files; focus on inspection, lookup, and summarization"
            } else {
                ""
            },
            if !history_enabled(input) {
                ""
            } else {
                "- Use `search_history`, `search_mem`, and related memory/search tools only when prior-turn recall is actually needed"
            }
        )),
    }
}

fn builtins_section(
    profile: PromptProfile,
    _tool_names: &[String],
    _history_enabled: bool,
    _memory_enabled: bool,
) -> String {
    let mut lines = vec![
        "## Runtime Globals".to_string(),
        "".to_string(),
        "- `observe expr`".to_string(),
        "- `finish expr`".to_string(),
    ];
    let _ = profile;
    lines.push("- Builtins: `len`, `empty`, `contains`, `slice`, `json_parse`, `format`, `to_string`, `to_int`, `to_float`".to_string());
    lines.join("\n")
}

fn render_guide_sections(sections: &[String]) -> String {
    sections
        .iter()
        .map(|section| section.trim())
        .filter(|section| !section.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn memory_section(is_subagent: bool, history_enabled: bool, memory_enabled: bool) -> String {
    let title = if history_enabled && memory_enabled {
        "## Memory & History"
    } else if memory_enabled {
        "## Memory"
    } else {
        "## History"
    };
    let mut lines = vec![title.to_string(), "".to_string()];
    if memory_enabled {
        lines.push(
            "- Memory is persistent across context pruning; use `mem_set` for durable decisions"
                .to_string(),
        );
        lines.push("- Retrieve memory with `mem_get`, `mem_all`, and `search_mem`".to_string());
    }
    if is_subagent {
        if history_enabled && memory_enabled {
            lines.push(
                "- History and memory can include parent-agent context inherited at spawn time"
                    .to_string(),
            );
        } else if history_enabled {
            lines.push(
                "- History can include parent-agent context inherited at spawn time".to_string(),
            );
        } else if memory_enabled {
            lines.push(
                "- Memory can include parent-agent context inherited at spawn time".to_string(),
            );
        }
    }
    lines.join("\n")
}

fn memory_api_section() -> String {
    [
        "## Memory API",
        "",
        "- `mem_set(key, description, value=None)` stores/updates memory",
        "- `mem_get(key)` returns `{key, description, value, turn}` or `null`; use `entry.value` to extract the stored value in REPL mode",
        "- `mem_delete(key)` removes a key",
        "- `mem_all()` lists keys with descriptions",
        "- `search_mem(query, mode=\"hybrid\", regex=None, limit=10, keys=None)` searches memory by content",
        "- Example: `call mem_set { key: \"repo_convention\", description: \"Use snake_case for tool names\" }`",
    ]
    .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::{
        AgentCapabilities, CapabilityId, helper_bindings_for_capability,
        prompt_sections_for_capabilities,
    };

    fn helpers_for(caps: &AgentCapabilities) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for id in &caps.enabled_capabilities {
            for helper in helper_bindings_for_capability(*id) {
                out.insert((*helper).to_string());
            }
        }
        out
    }

    fn prompt_sections_for(caps: &AgentCapabilities) -> Vec<String> {
        let available_tools = caps
            .enabled_capabilities
            .iter()
            .flat_map(|id| {
                crate::capabilities::tools_for_capability(*id)
                    .iter()
                    .copied()
            })
            .map(str::to_string)
            .collect::<BTreeSet<_>>();
        prompt_sections_for_capabilities(
            &caps.enabled_capabilities,
            &helpers_for(caps),
            &available_tools,
        )
    }

    fn can_write(caps: &AgentCapabilities) -> bool {
        caps.enabled(CapabilityId::CoreWrite)
    }

    #[test]
    fn parses_prompt_section_names() {
        assert_eq!(
            PromptSectionName::from_str("tool_guides").unwrap(),
            PromptSectionName::ToolGuides
        );
        assert_eq!(
            PromptSectionName::from_str("memory_api").unwrap(),
            PromptSectionName::MemoryApi
        );
        assert!(PromptSectionName::from_str("nope").is_err());
    }

    #[test]
    fn overrides_apply_in_order() {
        let overrides = vec![
            PromptSectionOverride {
                section: PromptSectionName::Identity,
                mode: PromptOverrideMode::Replace,
                content: "A".into(),
            },
            PromptSectionOverride {
                section: PromptSectionName::Identity,
                mode: PromptOverrideMode::Append,
                content: "B".into(),
            },
        ];
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::RootHeadless,
            execution_mode: crate::ExecutionMode::Repl,
            context: "ctx",
            tool_list: "tools",
            tool_names: &[],
            helper_bindings: &helpers_for(&AgentCapabilities::default()),
            guide_sections: &prompt_sections_for(&AgentCapabilities::default()),
            plugin_prompt_contributions: &[],
            can_write: can_write(&AgentCapabilities::default()),
            include_soul: false,
            project_instructions: "",
            overrides: &overrides,
        });
        assert!(text.starts_with("A\n\nB"));
    }

    #[test]
    fn memory_api_section_is_included_by_default() {
        let helper_bindings = BTreeSet::from([
            "search_mem".to_string(),
            "mem_set".to_string(),
            "mem_get".to_string(),
            "mem_delete".to_string(),
            "mem_all".to_string(),
        ]);
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::RootInteractive,
            execution_mode: crate::ExecutionMode::Repl,
            context: "ctx",
            tool_list: "tools",
            tool_names: &[],
            helper_bindings: &helper_bindings,
            guide_sections: &prompt_sections_for(&AgentCapabilities::default()),
            plugin_prompt_contributions: &[],
            can_write: can_write(&AgentCapabilities::default()),
            include_soul: false,
            project_instructions: "",
            overrides: &[],
        });
        assert!(text.contains("## Memory API"));
        assert!(text.contains("mem_set(key, description, value=None)"));
    }

    #[test]
    fn memory_api_section_can_be_disabled() {
        let overrides = vec![PromptSectionOverride {
            section: PromptSectionName::MemoryApi,
            mode: PromptOverrideMode::Disable,
            content: String::new(),
        }];
        let helper_bindings = BTreeSet::from([
            "search_mem".to_string(),
            "mem_set".to_string(),
            "mem_get".to_string(),
            "mem_delete".to_string(),
            "mem_all".to_string(),
        ]);
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::RootInteractive,
            execution_mode: crate::ExecutionMode::Repl,
            context: "ctx",
            tool_list: "tools",
            tool_names: &[],
            helper_bindings: &helper_bindings,
            guide_sections: &prompt_sections_for(&AgentCapabilities::default()),
            plugin_prompt_contributions: &[],
            can_write: can_write(&AgentCapabilities::default()),
            include_soul: false,
            project_instructions: "",
            overrides: &overrides,
        });
        assert!(!text.contains("## Memory API"));
    }

    #[test]
    fn repl_tool_access_describes_lashlang_calls() {
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::RootInteractive,
            execution_mode: crate::ExecutionMode::Repl,
            context: "ctx",
            tool_list: "tools",
            tool_names: &[
                "shell".to_string(),
                "agent_call".to_string(),
                "ask".to_string(),
            ],
            helper_bindings: &helpers_for(&AgentCapabilities::default()),
            guide_sections: &prompt_sections_for(&AgentCapabilities::default()),
            plugin_prompt_contributions: &[],
            can_write: can_write(&AgentCapabilities::default()),
            include_soul: false,
            project_instructions: "",
            overrides: &[],
        });
        assert!(text.contains("Call tools as `call tool_name { arg: expr }`"));
        assert!(text.contains("Use ternary expressions for inline branching: `cond ? yes : no`"));
        assert!(text.contains("Boolean negation supports both `!cond` and `not cond`"));
        assert!(text.contains("`format(value)` stringifies a single value"));
        assert!(text.contains("Prefer small REPL blocks over giant one-shot programs"));
        assert!(text.contains("inspect intermediate outputs"));
        assert!(text.contains("use `observe expr` for intermediate inspection"));
        assert!(text.contains("Only `observe expr` creates intermediate execution feedback"));
        assert!(text.contains("The user should see only final prose or `finish ...` output"));
        assert!(text.contains("- `observe expr`"));
        assert!(
            text.contains(
                "There is no `T` namespace, no imports, no methods, and no wrapper classes"
            )
        );
        assert!(text.contains("Use `parallel { ... }` for independent tool calls"));
        assert!(!text.contains("The runtime exposes only the listed tools"));
    }

    #[test]
    fn repl_prompt_explains_observe_vs_finish_in_interactive_mode() {
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::RootInteractive,
            execution_mode: crate::ExecutionMode::Repl,
            context: "ctx",
            tool_list: "tools",
            tool_names: &["search_tools".to_string()],
            helper_bindings: &helpers_for(&AgentCapabilities::default()),
            guide_sections: &prompt_sections_for(&AgentCapabilities::default()),
            plugin_prompt_contributions: &[],
            can_write: can_write(&AgentCapabilities::default()),
            include_soul: false,
            project_instructions: "",
            overrides: &[],
        });
        assert!(text.contains(
            "use `observe ...` to continue or `finish ...` when your final answer is ready"
        ));
        assert!(text.contains("`finish ...` ends the turn; otherwise the turn may continue"));
        assert!(text.contains("## Available Tools"));
    }

    #[test]
    fn skills_not_mentioned_when_skills_capability_disabled() {
        let caps = AgentCapabilities::default().disable(CapabilityId::Skills);
        let tools = vec!["load_skill".to_string(), "search_skills".to_string()];
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::RootInteractive,
            execution_mode: crate::ExecutionMode::Repl,
            context: "ctx",
            tool_list: "tools",
            tool_names: &tools,
            helper_bindings: &helpers_for(&caps),
            guide_sections: &prompt_sections_for(&caps),
            plugin_prompt_contributions: &[],
            can_write: can_write(&caps),
            include_soul: false,
            project_instructions: "",
            overrides: &[],
        });
        assert!(!text.contains("search_skills("));
        assert!(!text.contains("### Skills"));
    }

    #[test]
    fn guide_sections_render_inside_tool_guide_section() {
        let guides = vec![
            "### First Guide\nFirst details.".to_string(),
            "### Second Guide\nSecond details.".to_string(),
        ];
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::RootInteractive,
            execution_mode: crate::ExecutionMode::Repl,
            context: "ctx",
            tool_list: "tools",
            tool_names: &[],
            helper_bindings: &helpers_for(&AgentCapabilities::default()),
            guide_sections: &guides,
            plugin_prompt_contributions: &[],
            can_write: can_write(&AgentCapabilities::default()),
            include_soul: false,
            project_instructions: "",
            overrides: &[],
        });
        assert!(text.contains("## Tool Guide"));
        assert!(text.contains("### First Guide"));
        assert!(text.contains("Second details."));
    }

    #[test]
    fn subagent_prompt_mentions_read_only_when_core_write_disabled() {
        let caps = AgentCapabilities::default().disable(CapabilityId::CoreWrite);
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::SubAgentInteractive,
            execution_mode: crate::ExecutionMode::Repl,
            context: "ctx",
            tool_list: "tools",
            tool_names: &[],
            helper_bindings: &helpers_for(&caps),
            guide_sections: &prompt_sections_for(&caps),
            plugin_prompt_contributions: &[],
            can_write: can_write(&caps),
            include_soul: false,
            project_instructions: "",
            overrides: &[],
        });
        assert!(text.contains("read-only sub-agent"));
        assert!(text.contains("This agent is read-only"));
    }

    #[test]
    fn dynamic_capability_prompt_sections_are_included() {
        let caps = AgentCapabilities::default();
        let sections = vec!["## Custom Capability\n\nCustom guidance.".to_string()];
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::RootInteractive,
            execution_mode: crate::ExecutionMode::Repl,
            context: "ctx",
            tool_list: "tools",
            tool_names: &["agent_call".to_string()],
            helper_bindings: &helpers_for(&caps),
            guide_sections: &sections,
            plugin_prompt_contributions: &[],
            can_write: can_write(&caps),
            include_soul: false,
            project_instructions: "",
            overrides: &[],
        });
        assert!(text.contains("## Custom Capability"));
        assert!(text.contains("Custom guidance."));
    }

    #[test]
    fn standard_prompt_emphasizes_parallel_independent_calls() {
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::RootInteractive,
            execution_mode: crate::ExecutionMode::Standard,
            context: "ctx",
            tool_list: "tools",
            tool_names: &[],
            helper_bindings: &helpers_for(&AgentCapabilities::default()),
            guide_sections: &prompt_sections_for(&AgentCapabilities::default()),
            plugin_prompt_contributions: &[],
            can_write: can_write(&AgentCapabilities::default()),
            include_soul: false,
            project_instructions: "",
            overrides: &[],
        });
        assert!(text.contains("Group independent tool calls in the same response"));
        assert!(text.contains("Good fits: reading several files"));
        assert!(text.contains("Do not parallelize dependent steps"));
    }

    #[test]
    fn standard_prompt_omits_repl_runtime_globals() {
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::RootInteractive,
            execution_mode: crate::ExecutionMode::Standard,
            context: "ctx",
            tool_list: "tools",
            tool_names: &["ask".to_string()],
            helper_bindings: &helpers_for(&AgentCapabilities::default()),
            guide_sections: &prompt_sections_for(&AgentCapabilities::default()),
            plugin_prompt_contributions: &[],
            can_write: can_write(&AgentCapabilities::default()),
            include_soul: false,
            project_instructions: "",
            overrides: &[],
        });
        assert!(!text.contains("## Runtime Globals"));
        assert!(!text.contains("reset_repl()"));
        assert!(!text.contains("done(value)"));
        assert!(text.contains("Use only tools shown in Available Tools"));
    }

    #[test]
    fn prompt_strengthens_completion_and_validation_rules() {
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::RootInteractive,
            execution_mode: crate::ExecutionMode::Standard,
            context: "ctx",
            tool_list: "tools",
            tool_names: &[],
            helper_bindings: &helpers_for(&AgentCapabilities::default()),
            guide_sections: &prompt_sections_for(&AgentCapabilities::default()),
            plugin_prompt_contributions: &[],
            can_write: can_write(&AgentCapabilities::default()),
            include_soul: false,
            project_instructions: "",
            overrides: &[],
        });
        assert!(text.contains("Keep going until the task is complete"));
        assert!(text.contains("inspect or validate more instead of guessing"));
        assert!(text.contains("Validate the smallest relevant thing first"));
        assert!(text.contains("Do not fix unrelated failures uncovered during validation"));
    }

    #[test]
    fn prompt_orders_dynamic_sections_late() {
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::RootInteractive,
            execution_mode: crate::ExecutionMode::Repl,
            context: "ctx",
            tool_list: "tools",
            tool_names: &["read_file".to_string()],
            helper_bindings: &helpers_for(&AgentCapabilities::default()),
            guide_sections: &prompt_sections_for(&AgentCapabilities::default()),
            plugin_prompt_contributions: &[],
            can_write: can_write(&AgentCapabilities::default()),
            include_soul: true,
            project_instructions: "project rules",
            overrides: &[],
        });
        let guidelines_idx = text.find("## Guidelines").unwrap();
        let project_idx = text.find("## Project Instructions").unwrap();
        let env_idx = text.find("## Environment").unwrap();
        let tools_idx = text.find("## Available Tools").unwrap();
        assert!(guidelines_idx < project_idx);
        assert!(project_idx < env_idx);
        assert!(env_idx < tools_idx);
    }

    #[test]
    fn prompt_does_not_emit_runtime_prune_status() {
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::RootInteractive,
            execution_mode: crate::ExecutionMode::Repl,
            context: "ctx",
            tool_list: "tools",
            tool_names: &[],
            helper_bindings: &helpers_for(&AgentCapabilities::default()),
            guide_sections: &prompt_sections_for(&AgentCapabilities::default()),
            plugin_prompt_contributions: &[],
            can_write: can_write(&AgentCapabilities::default()),
            include_soul: false,
            project_instructions: "",
            overrides: &[],
        });
        assert!(!text.contains("Context-pruned turns this run"));
        assert!(!text.contains("Skip history-mining detours"));
    }
}
