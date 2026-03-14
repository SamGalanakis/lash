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
        SECTION_DEFS
            .iter()
            .map(|def| def.name.as_str())
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
    Root,
    SubAgent,
}

impl PromptProfile {
    pub fn from_subagent(sub_agent: bool) -> Self {
        if sub_agent {
            Self::SubAgent
        } else {
            Self::Root
        }
    }

    pub fn is_subagent(self) -> bool {
        matches!(self, Self::SubAgent)
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
                "search_mem" | "mem_set" | "mem_get" | "mem_delete"
            )
        })
}

type PromptSectionBuilder = for<'a> fn(&PromptRenderContext<'a>) -> Option<String>;

struct PromptSectionDef {
    name: PromptSectionName,
    builder: PromptSectionBuilder,
}

const SECTION_DEFS: [PromptSectionDef; 15] = [
    PromptSectionDef {
        name: PromptSectionName::Identity,
        builder: identity_section,
    },
    PromptSectionDef {
        name: PromptSectionName::Personality,
        builder: personality_section,
    },
    PromptSectionDef {
        name: PromptSectionName::ExecutionContract,
        builder: execution_contract_section,
    },
    PromptSectionDef {
        name: PromptSectionName::TerminationContract,
        builder: termination_contract_section,
    },
    PromptSectionDef {
        name: PromptSectionName::ErrorRecovery,
        builder: error_recovery_section,
    },
    PromptSectionDef {
        name: PromptSectionName::Guidelines,
        builder: guidelines_section,
    },
    PromptSectionDef {
        name: PromptSectionName::ProjectInstructions,
        builder: project_instructions_section,
    },
    PromptSectionDef {
        name: PromptSectionName::Environment,
        builder: environment_section,
    },
    PromptSectionDef {
        name: PromptSectionName::Memory,
        builder: memory_prompt_section,
    },
    PromptSectionDef {
        name: PromptSectionName::MemoryApi,
        builder: memory_api_prompt_section,
    },
    PromptSectionDef {
        name: PromptSectionName::ToolAccess,
        builder: tool_access_section,
    },
    PromptSectionDef {
        name: PromptSectionName::Builtins,
        builder: builtins_prompt_section,
    },
    PromptSectionDef {
        name: PromptSectionName::ToolGuides,
        builder: tool_guides_section,
    },
    PromptSectionDef {
        name: PromptSectionName::PluginExtensions,
        builder: plugin_extensions_section,
    },
    PromptSectionDef {
        name: PromptSectionName::AvailableTools,
        builder: available_tools_section,
    },
];

struct PromptRenderContext<'a> {
    input: &'a PromptComposeInput<'a>,
    is_repl: bool,
    is_subagent: bool,
    history_enabled: bool,
    memory_enabled: bool,
}

impl<'a> PromptRenderContext<'a> {
    fn new(input: &'a PromptComposeInput<'a>) -> Self {
        Self {
            input,
            is_repl: matches!(input.execution_mode, ExecutionMode::Repl),
            is_subagent: input.profile.is_subagent(),
            history_enabled: history_enabled(input),
            memory_enabled: memory_enabled(input),
        }
    }
}

pub fn compose_system_prompt(input: PromptComposeInput<'_>) -> String {
    let render_context = PromptRenderContext::new(&input);
    let mut sections: HashMap<PromptSectionName, Option<String>> = HashMap::new();
    for def in SECTION_DEFS {
        sections.insert(def.name, (def.builder)(&render_context));
    }
    apply_prompt_contributions(&mut sections, input.plugin_prompt_contributions);
    apply_overrides(&mut sections, input.overrides);

    SECTION_DEFS
        .iter()
        .filter_map(|def| sections.get(&def.name).cloned().flatten())
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

fn identity_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    let input = ctx.input;
    Some(match input.profile {
        PromptProfile::Root => {
            if ctx.is_repl {
                "You are an AI coding assistant operating in a persistent REPL with tool access.\nYou power lash, a terminal-based coding agent. Understand the codebase, make changes, run commands, and report outcomes clearly.".to_string()
            } else {
                "You are an AI coding assistant with direct tool-calling access.\nYou power lash, a terminal-based coding agent. Understand the codebase, make changes, run commands, and report outcomes clearly.".to_string()
            }
        }
        PromptProfile::SubAgent => {
            if ctx.is_repl {
                if input.can_write {
                    "You are a sub-agent inside lash working on a delegated task.\nUse tools decisively and return results to the caller via `finish ...` when complete.".to_string()
                } else {
                    "You are a read-only sub-agent inside lash working on a delegated task.\nFocus on lookup/summarization tasks and return results to the caller via `finish ...` when complete.".to_string()
                }
            } else if input.can_write {
                "You are a sub-agent inside lash working on a delegated task.\nUse tools decisively and return a final answer to the caller when complete.".to_string()
            } else {
                "You are a read-only sub-agent inside lash working on a delegated task.\nFocus on lookup/summarization tasks and return a final answer to the caller when complete.".to_string()
            }
        }
    })
}

fn environment_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    Some(format!("## Environment\n\n{}", ctx.input.context))
}

fn personality_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    if !ctx.input.include_soul {
        None
    } else {
        Some(
            "## Core Principles\n\n- First-principles thinker\n- Allergic to accidental complexity\n- Direct over diplomatic\n- Skeptical of abstraction\n- Show, don't lecture\n- High standards by default"
                .to_string(),
        )
    }
}

fn execution_contract_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    Some(format!(
        "{}\n{}",
        if ctx.is_repl {
            "## Execution Contract\n\nYour output can include prose and `<repl>` blocks.\n- Use prose only when no execution is needed\n- `<repl>` executes immediately when `</repl>` is reached\n- `finish expr` ends the turn, even when the value renders as empty text\n- `observe expr` does not end the turn; it sends hidden feedback to the next model step\n- A `<repl>` block may also do work silently and continue without `finish` or `observe`\n- Maximum one `<repl>` block per response\n- For direct conversational requests that need no tools, respond in prose only\n- The REPL has no `print`; use `observe expr` for intermediate inspection and `finish ...` for the final answer\n- Prefer small REPL blocks over giant one-shot programs\n- Validate one or two steps, inspect intermediate outputs, then extend the workflow\n- Only `observe expr` creates intermediate execution feedback for the next model step; ordinary assignments are silent\n- The user should see only final prose or `finish ...` output, not intermediate observations\n\n### REPL Language\n\nThe REPL is `lashlang`, a small workflow language for tool orchestration.\n- Values are null, booleans, numbers, strings, lists, and records\n- Assign with `name = expr`\n- Bare expressions are valid statements; in `parallel { ... }`, a bare expression branch contributes that value to the result list\n- Call tools with `call tool_name { arg: expr }`\n- Use `observe expr` to inspect a value and continue execution\n- Use `finish expr` to end the turn and return a final answer\n- Control flow is limited to statement `if`, `for`, and `finish`; `parallel` also works as an expression\n- `parallel { ... }` in expression position returns a list of branch results\n- Use ternary expressions for inline branching: `cond ? yes : no`\n- Boolean negation supports both `!cond` and `not cond`\n- Boolean conjunction/disjunction support both `&&` / `||` and `and` / `or`\n- Tool results are records like `{ ok: true, value: ... }` or `{ ok: false, error: ... }`\n- Access the wrapped payload via `.value` only when `result.ok` is true\n- Do not assume every `value` is a record: many tools return strings, numbers, or lists directly\n- There are no imports, classes, methods, exceptions, comprehensions, or arbitrary standard library access\n- Use builtins like `len(...)`, `empty(...)`, `contains(...)`, `slice(...)`, `json_parse(...)`, `format(...)`, and `to_string(...)`\n- Builtins return plain values; invalid builtin usage raises a runtime error instead of returning a `{ ok, error }` record\n- `slice(value, start, end)` treats `null` bounds as omitted: `start=null` means from the beginning, `end=null` means through the end\n- `to_string(value)` stringifies a single value\n- `format(\"...\", args...)` formats templates with `{}` placeholders; use `{0}`, `{1}`, ... only when argument reordering matters\n- Escape literal braces in templates with `{{` and `}}`\n- String `+` concatenation auto-stringifies when either side is already a string\n- If you need unsupported features, use the appropriate host tool instead of emulating them inside the REPL"
        } else {
            "## Execution Contract\n\nUse direct tool calls when execution is needed.\n- Do not emit `<repl>` blocks or Python code\n- Call tools directly with valid arguments\n- You may call multiple tools in a single response\n- When tool calls are independent, emit them together in the same response so the runtime can execute them concurrently\n- Prefer native parallel tool calls aggressively for unrelated reads, searches, and diagnostics\n- Serialize only when later arguments depend on earlier results\n- Avoid filler prose between tool calls\n- Keep going until the task is complete; do not stop after inspection or partial progress\n- If you are unsure, inspect or validate more instead of guessing\n- For direct conversational requests that need no tools, respond in prose only"
        },
        if ctx.is_repl {
            "- In interactive mode, prose-only is fine only if you never opened `<repl>`; after REPL execution, use `observe ...` to continue or `finish ...` when your final answer is ready. If you want user-visible commentary after an `observe`, put it at the start of the next response before the next `<repl>` block."
        } else {
            "- In interactive mode, return a final user-facing answer once the task is complete."
        }
    ))
}

fn termination_contract_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    Some(format!(
        "{}\n{}",
        if ctx.is_repl {
            "## Termination Contract\n\n`finish expr` ends the turn and returns control.\n- `finish ...` may only be used inside `<repl>`\n- If you never used `<repl>` in the turn, you may finish with plain prose instead\n- If you used `<repl>` at any point in the turn, only `finish ...` ends the turn\n- Do not use `finish ...` for status updates"
        } else {
            "## Termination Contract\n\nFinish by returning a final assistant answer.\n- Do not emit fake tool calls or placeholder arguments\n- Do not stop after tool execution unless the task is actually complete"
        },
        if ctx.is_repl {
            "- Interactive: `finish ...` ends the turn; otherwise the turn may continue after `observe ...` or silent REPL work"
        } else {
            "- Interactive: provide a concise final answer when complete."
        }
    ))
}

fn tool_access_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    Some(if ctx.is_repl {
        "## Tool Access\n\n- Call tools as `call tool_name { arg: expr }`\n- Tool results are wrapped records; read `result.ok`, `result.value`, and `result.error`\n- Structured payload fields live directly under `result.value`; do not invent extra wrapper names like `result.value.path_entries`\n- There is no `T` namespace, no imports, no methods, and no wrapper classes; work with plain records/lists/primitives\n- Use `parallel { ... }` for independent tool calls\n- `parallel { ... }` returns a list of branch results in source order\n- Branches that end with `call ...` produce the same wrapped `{ ok, value, error }` records as ordinary tool calls"
            .to_string()
    } else {
        "## Tool Access\n\n- The runtime exposes only the listed tools\n- Use only tools shown in Available Tools\n- Read each tool's description and examples before calling it\n- When several listed tools are independent, call them together in one response instead of serializing them\n- Never invent tool names or arguments".to_string()
    })
}

fn tool_guides_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    let guides = render_guide_sections(ctx.input.guide_sections);
    if guides.is_empty() {
        None
    } else {
        Some(format!("## Tool Guide\n\n{}", guides))
    }
}

fn available_tools_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    Some(format!("## Available Tools\n\n{}", ctx.input.tool_list))
}

fn error_recovery_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    Some(if ctx.is_repl {
        "## Error Recovery\n\nTool failures return error records from `call`.\n- Inspect `result.ok` / `result.error` and fix the cause before retrying\n- Builtins do not return error records; invalid builtin usage raises a runtime error that aborts the current REPL block\n- Do not repeat failing calls unchanged\n- If a REPL program fails, shrink the block, use `observe expr` on the next uncertain value, and rerun a smaller `<repl>` block before expanding again\n- When a workflow is uncertain, validate the next small step instead of writing the whole program at once"
            .to_string()
    } else {
        "## Error Recovery\n\nTool failures return structured errors.\n- Read the error carefully and fix the cause before retrying\n- Do not repeat failing calls unchanged\n- If a tool result is incomplete or ambiguous, inspect with other tools instead of guessing"
            .to_string()
    })
}

fn builtins_prompt_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    if ctx.is_repl {
        Some(builtins_section(
            ctx.input.profile,
            ctx.input.tool_names,
            ctx.history_enabled,
            ctx.memory_enabled,
        ))
    } else {
        None
    }
}

fn memory_prompt_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    if ctx.history_enabled || ctx.memory_enabled {
        Some(memory_section(
            ctx.is_subagent,
            ctx.history_enabled,
            ctx.memory_enabled,
        ))
    } else {
        None
    }
}

fn memory_api_prompt_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    if ctx.memory_enabled {
        Some(memory_api_section())
    } else {
        None
    }
}

fn project_instructions_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    if ctx.input.project_instructions.trim().is_empty() {
        None
    } else {
        Some(format!(
            "## Project Instructions\n\n{}",
            ctx.input.project_instructions
        ))
    }
}

fn plugin_extensions_section(_ctx: &PromptRenderContext<'_>) -> Option<String> {
    None
}

fn guidelines_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    Some(format!(
        "## Guidelines\n\n- Bias toward concrete execution over planning chatter\n- Keep going until the request is resolved; do not stop at reconnaissance when a concrete deliverable is requested\n- Validate the smallest relevant thing first, then broaden if needed\n- Do not fix unrelated failures uncovered during validation; report them instead\n- For substantial scripts/workflows, create files and run them with host tooling\n- Use isolated environments only when required dependencies are missing\n- Avoid redundant file reads when values already exist in variables\n- Never speculate about files you have not read\n- Be concise and action-oriented\n{}\n{}",
        if !ctx.input.can_write {
            "- This agent is read-only: do not modify files; focus on inspection, lookup, and summarization"
        } else {
            ""
        },
        if !ctx.history_enabled {
            ""
        } else {
            "- Use `search_history`, `search_mem`, and related memory/search tools only when prior-turn recall is actually needed"
        }
    ))
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
        lines.push("- Retrieve memory with `mem_get` and `search_mem`".to_string());
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
        "- `search_mem(query=None, mode=\"hybrid\", regex=None, limit=None, keys=None)` searches memory by content; with no query, it lists all memory in stable key order",
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
            profile: PromptProfile::Root,
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
        ]);
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::Root,
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
        ]);
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::Root,
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
            profile: PromptProfile::Root,
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
        assert!(text.contains("`to_string(value)` stringifies a single value"));
        assert!(
            text.contains("`format(\"...\", args...)` formats templates with `{}` placeholders")
        );
        assert!(text.contains("Escape literal braces in templates with `{{` and `}}`"));
        assert!(text.contains(
            "Builtins return plain values; invalid builtin usage raises a runtime error"
        ));
        assert!(text.contains(
            "Builtins do not return error records; invalid builtin usage raises a runtime error"
        ));
        assert!(text.contains("Prefer small REPL blocks over giant one-shot programs"));
        assert!(text.contains("inspect intermediate outputs"));
        assert!(text.contains("use `observe expr` for intermediate inspection"));
        assert!(text.contains("Only `observe expr` creates intermediate execution feedback"));
        assert!(text.contains("The user should see only final prose or `finish ...` output"));
        assert!(text.contains("- `observe expr`"));
        assert!(text.contains("`parallel` also works as an expression"));
        assert!(text.contains("returns a list of branch results"));
        assert!(
            text.contains(
                "There is no `T` namespace, no imports, no methods, and no wrapper classes"
            )
        );
        assert!(text.contains("Use `parallel { ... }` for independent tool calls"));
        assert!(text.contains("returns a list of branch results in source order"));
        assert!(text.contains("same wrapped `{ ok, value, error }` records"));
        assert!(!text.contains("The runtime exposes only the listed tools"));
    }

    #[test]
    fn repl_prompt_explains_observe_vs_finish_in_interactive_mode() {
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::Root,
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
        assert!(
            text.contains(
                "put it at the start of the next response before the next `<repl>` block"
            )
        );
        assert!(!text.contains("Do not assume prose after `</repl>` is user-visible"));
        assert!(text.contains("`finish ...` ends the turn; otherwise the turn may continue"));
        assert!(text.contains("## Available Tools"));
    }

    #[test]
    fn skills_not_mentioned_when_skills_capability_disabled() {
        let caps = AgentCapabilities::default().disable(CapabilityId::Skills);
        let tools = vec!["load_skill".to_string(), "search_skills".to_string()];
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::Root,
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
            profile: PromptProfile::Root,
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
            profile: PromptProfile::SubAgent,
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
            profile: PromptProfile::Root,
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
    fn standard_prompt_describes_native_parallel_tool_calls() {
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::Root,
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
        assert!(text.contains("Read each tool's description and examples before calling it"));
        assert!(text.contains("You may call multiple tools in a single response"));
        assert!(text.contains("call them together in one response instead of serializing them"));
        assert!(!text.contains("batch(calls=["));
    }

    #[test]
    fn standard_prompt_omits_repl_runtime_globals() {
        let text = compose_system_prompt(PromptComposeInput {
            profile: PromptProfile::Root,
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
            profile: PromptProfile::Root,
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
            profile: PromptProfile::Root,
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
            profile: PromptProfile::Root,
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
