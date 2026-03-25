use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;

use crate::PromptContext;
use crate::plugin::PromptContribution;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptSectionName {
    Intro,
    Execution,
    Guidance,
    Environment,
    AvailableTools,
}

impl PromptSectionName {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Intro => "intro",
            Self::Execution => "execution",
            Self::Guidance => "guidance",
            Self::Environment => "environment",
            Self::AvailableTools => "available_tools",
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
            "intro" => Ok(Self::Intro),
            "execution" => Ok(Self::Execution),
            "guidance" => Ok(Self::Guidance),
            "environment" => Ok(Self::Environment),
            "available_tools" => Ok(Self::AvailableTools),
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

pub trait PromptRenderer: Send + Sync {
    fn render(&self, prompt: &PromptContext, overrides: &[PromptSectionOverride]) -> String;
}

type PromptSectionBuilder = for<'a> fn(&PromptRenderContext<'a>) -> Option<String>;

struct PromptSectionDef {
    name: PromptSectionName,
    builder: PromptSectionBuilder,
}

const SECTION_DEFS: [PromptSectionDef; 5] = [
    PromptSectionDef {
        name: PromptSectionName::Intro,
        builder: intro_section,
    },
    PromptSectionDef {
        name: PromptSectionName::Execution,
        builder: execution_section,
    },
    PromptSectionDef {
        name: PromptSectionName::Guidance,
        builder: guidance_section,
    },
    PromptSectionDef {
        name: PromptSectionName::Environment,
        builder: environment_section,
    },
    PromptSectionDef {
        name: PromptSectionName::AvailableTools,
        builder: available_tools_section,
    },
];

struct PromptRenderContext<'a> {
    prompt: &'a PromptContext,
    is_repl: bool,
}

impl<'a> PromptRenderContext<'a> {
    fn new(prompt: &'a PromptContext) -> Self {
        Self {
            prompt,
            is_repl: matches!(prompt.mode, crate::ExecutionMode::Repl),
        }
    }
}

#[derive(Default)]
pub struct DefaultPromptRenderer;

impl PromptRenderer for DefaultPromptRenderer {
    fn render(&self, prompt: &PromptContext, overrides: &[PromptSectionOverride]) -> String {
        let render_context = PromptRenderContext::new(prompt);
        let mut sections: HashMap<PromptSectionName, Option<String>> = HashMap::new();
        for def in SECTION_DEFS {
            sections.insert(def.name, (def.builder)(&render_context));
        }
        apply_prompt_contributions(&mut sections, &prompt.contributions);
        apply_overrides(&mut sections, overrides);

        SECTION_DEFS
            .iter()
            .filter_map(|def| sections.get(&def.name).cloned().flatten())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

pub fn default_prompt_renderer() -> Arc<dyn PromptRenderer> {
    Arc::new(DefaultPromptRenderer)
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

fn intro_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    let mut blocks = vec![if ctx.prompt.is_subagent {
        if ctx.prompt.can_write {
            "You are a sub-agent inside lash working on a delegated task.\nUse tools decisively and return results to the caller when complete.".to_string()
        } else {
            "You are a read-only sub-agent inside lash working on a delegated task.\nFocus on lookup and summarization work, then return results to the caller.".to_string()
        }
    } else {
        "You are an AI coding assistant operating inside lash with tool access.\nUnderstand the codebase, make changes, run commands, and report outcomes clearly.".to_string()
    }];

    if ctx.prompt.include_soul {
        blocks.push(
            "## Core Principles\n\n- First-principles thinker\n- Allergic to accidental complexity\n- Direct over diplomatic\n- Skeptical of abstraction\n- Show, don't lecture\n- High standards by default"
                .to_string(),
        );
    }

    Some(blocks.join("\n\n"))
}

fn execution_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    Some(if ctx.is_repl {
        "## Execution\n\nYour output can include prose and `<repl>` blocks.\n- Work iteratively: inspect, act, observe, continue\n- Most tasks take multiple REPL cycles, not one large block\n- Use at most one `<repl>` block per response; once you close `</repl>`, stop and wait for the result\n- If you need tools or execution, emit a `<repl>` block and stop there\n- If the task is complete, do not emit `<repl>`; reply in plain prose and that finalizes the turn\n- Never put user-facing prose after `</repl>`; anything after the first closed block will be ignored\n- Use `observe` for intermediate results, inspection, and progress that should continue; `observe` output is hidden from the user\n- Verify the concrete end state before replying in prose when possible\n\n### REPL Language\n\nThe REPL is `lashlang`, a small workflow language for tool orchestration.\n- Values are null, booleans, numbers, strings, lists, and records\n- List and record literals use comma-separated entries: `[a, b]`, `{ a: 1, b: 2 }`; tool arg records follow the same rule\n- Assign with `name = expr`\n- Bare expressions are valid statements; in `parallel { ... }`, a bare expression branch contributes that value to the result list\n- Call tools with `call tool_name { arg: expr }`\n- Use `parallel { ... }` only for independent tool calls; if one call needs another call's output, do not put them in the same `parallel { ... }`\n- `parallel { ... }` returns a list of branch results in source order, and branches that end with `call ...` produce the same wrapped `{ ok, value, error }` records as ordinary tool calls\n- Use `observe expr` to inspect a value and continue execution\n- Control flow is limited to statement `if` and `for`; `parallel` also works as an expression\n- Use ternary expressions for inline branching: `cond ? yes : no`\n- Boolean negation supports both `!cond` and `not cond`\n- Boolean conjunction/disjunction support both `&&` / `||` and `and` / `or`\n- Tool results are records like `{ ok: true, value: ... }` or `{ ok: false, error: ... }`\n- Access the wrapped payload via `.value` only when `result.ok` is true\n- Do not assume every `value` is a record: many tools return strings, numbers, or lists directly\n- Builtins: `len`, `empty`, `contains`, `slice`, `json_parse`, `format`, `to_string`, `to_int`, `to_float`\n- Builtins return plain values; invalid builtin usage raises a runtime error instead of returning a `{ ok, error }` record\n- `slice(value, start, end)` treats `null` bounds as omitted: `start=null` means from the beginning, `end=null` means through the end\n- `to_string(value)` stringifies a single value\n- `format(\"...\", args...)` formats templates with `{}` placeholders; use `{0}`, `{1}`, ... only when argument reordering matters\n- Escape literal braces in templates with `{{` and `}}`\n- String `+` concatenation auto-stringifies when either side is already a string".to_string()
    } else {
        "## Execution\n\nUse direct tool calls when execution is needed.\n- Do not emit `<repl>` blocks or Python code\n- Call tools directly with valid arguments\n- Use `batch` for 2 or more independent tool calls; serialize only when later arguments depend on earlier results\n- Avoid filler prose between tool calls\n- Work in small, concrete steps and verify each meaningful step before broadening scope\n- After edits, run the narrowest check that can falsify the change before moving on to broader validation\n- If a tool fails or returns incomplete output, inspect the current state, fix the cause, and continue; do not repeat the same failing call unchanged\n- Keep going until the task is complete; do not stop after inspection or partial progress\n- If you are unsure, resolve the uncertainty with the smallest relevant check; broaden only when the current path is insufficient\n- Before concluding, verify the concrete end-state with tools whenever possible\n- For direct conversational requests that need no tools, respond in prose only\n- Finish by returning a final assistant answer once the task is actually complete\n- In interactive mode, return a concise final user-facing answer when complete".to_string()
    })
}

fn guidance_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    Some(format!(
        "## Guidance\n\n- {}read each tool's description and examples when provided before calling it\n- Tool and plugin sections own the syntax, examples, and special affordances for their surfaces\n- Bias toward concrete execution over abstract discussion\n- Keep going until the request is resolved; do not stop at reconnaissance when a concrete deliverable is requested\n- Validate the smallest relevant thing first, then broaden if needed\n- Prefer the next concrete step that reduces uncertainty\n- Do not fix unrelated failures uncovered during validation; report them instead\n- Never invent tool names or arguments\n- For substantial scripts or workflows, create files and run them with host tooling\n- Use isolated environments only when required dependencies are missing\n- Avoid redundant file reads when values already exist in variables\n- Never speculate about files you have not read\n- Be concise and action-oriented{}\n",
        if ctx.is_repl {
            "Use only tools shown in Available Tools; "
        } else {
            "Use only available tools; "
        },
        if ctx.prompt.can_write {
            ""
        } else {
            "\n- This agent is read-only: do not modify files; focus on inspection, lookup, and summarization"
        }
    ))
}

fn environment_section(_ctx: &PromptRenderContext<'_>) -> Option<String> {
    None
}

fn available_tools_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    if ctx.is_repl {
        Some(format!("## Available Tools\n\n{}", ctx.prompt.tool_list))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prompt(mode: crate::ExecutionMode) -> PromptContext {
        PromptContext {
            mode,
            tool_list: "tools".to_string(),
            include_soul: false,
            ..PromptContext::default()
        }
    }

    #[test]
    fn parses_prompt_section_names() {
        assert_eq!(
            PromptSectionName::from_str("available_tools").unwrap(),
            PromptSectionName::AvailableTools
        );
        assert_eq!(
            PromptSectionName::from_str("guidance").unwrap(),
            PromptSectionName::Guidance
        );
        assert!(PromptSectionName::from_str("memory_api").is_err());
    }

    #[test]
    fn overrides_apply_in_order() {
        let overrides = vec![
            PromptSectionOverride {
                section: PromptSectionName::Intro,
                mode: PromptOverrideMode::Replace,
                content: "A".into(),
            },
            PromptSectionOverride {
                section: PromptSectionName::Intro,
                mode: PromptOverrideMode::Append,
                content: "B".into(),
            },
        ];
        let text = DefaultPromptRenderer.render(&prompt(crate::ExecutionMode::Repl), &overrides);
        assert!(text.starts_with("A\n\nB"));
    }

    #[test]
    fn repl_prompt_keeps_lashlang_contract() {
        let text = DefaultPromptRenderer.render(&prompt(crate::ExecutionMode::Repl), &[]);
        assert!(text.contains("Call tools with `call tool_name { arg: expr }`"));
        assert!(text.contains(
            "List and record literals use comma-separated entries: `[a, b]`, `{ a: 1, b: 2 }`"
        ));
        assert!(text.contains("Use ternary expressions for inline branching: `cond ? yes : no`"));
        assert!(text.contains("Boolean negation supports both `!cond` and `not cond`"));
        assert!(text.contains("Use `observe expr` to inspect a value and continue execution"));
        assert!(text.contains("Work iteratively: inspect, act, observe, continue"));
        assert!(text.contains("Use at most one `<repl>` block per response"));
        assert!(text.contains("If the task is complete, do not emit `<repl>`"));
        assert!(text.contains("Control flow is limited to statement `if` and `for`"));
        assert!(!text.contains("finish"));
        assert!(text.contains("## Available Tools"));
    }

    #[test]
    fn standard_prompt_strengthens_validation_without_tool_specific_noise() {
        let text = DefaultPromptRenderer.render(&prompt(crate::ExecutionMode::Standard), &[]);
        assert!(text.contains("Work in small, concrete steps and verify each meaningful step"));
        assert!(text.contains("Before concluding, verify the concrete end-state"));
        assert!(text.contains("resolve the uncertainty with the smallest relevant check"));
        assert!(!text.contains("inspect or validate more instead of guessing"));
        assert!(text.contains(
            "Tool and plugin sections own the syntax, examples, and special affordances"
        ));
        assert!(text.contains("Use only available tools; read each tool's description"));
        assert!(!text.contains("update_plan"));
        assert!(!text.contains("## Tool Access"));
        assert!(!text.contains("## Memory API"));
        assert!(!text.contains("## Available Tools"));
    }

    #[test]
    fn guidance_sections_render_inside_guidance() {
        let mut prompt = prompt(crate::ExecutionMode::Repl);
        prompt.contributions = vec![
            PromptContribution::guidance("### First Guide\nFirst details."),
            PromptContribution::guidance("### Second Guide\nSecond details."),
        ];
        let text = DefaultPromptRenderer.render(&prompt, &[]);
        assert!(text.contains("## Guidance"));
        assert!(text.contains("### First Guide"));
        assert!(text.contains("Second details."));
    }

    #[test]
    fn prompt_contributions_append_to_sections() {
        let mut prompt = prompt(crate::ExecutionMode::Repl);
        prompt.contributions = vec![
            PromptContribution {
                section: PromptSectionName::Environment,
                priority: 0,
                content: "cwd: /tmp/demo".to_string(),
            },
            PromptContribution {
                section: PromptSectionName::Guidance,
                priority: 0,
                content: "### Project Instructions\nDo the safe thing.".to_string(),
            },
        ];
        let text = DefaultPromptRenderer.render(&prompt, &[]);
        assert!(text.contains("cwd: /tmp/demo"));
        assert!(text.contains("### Project Instructions"));
    }

    #[test]
    fn subagent_prompt_mentions_read_only_when_writes_disabled() {
        let mut prompt = prompt(crate::ExecutionMode::Repl);
        prompt.is_subagent = true;
        prompt.can_write = false;
        let text = DefaultPromptRenderer.render(&prompt, &[]);
        assert!(text.contains("read-only sub-agent"));
        assert!(text.contains("This agent is read-only"));
    }

    #[test]
    fn prompt_orders_sections_late_for_environment_and_tools() {
        let mut prompt = prompt(crate::ExecutionMode::Repl);
        prompt.include_soul = true;
        prompt.contributions = vec![
            PromptContribution::guidance("### Custom\nMore guidance."),
            PromptContribution {
                section: PromptSectionName::Environment,
                priority: 0,
                content: "cwd: /repo".to_string(),
            },
        ];
        let text = DefaultPromptRenderer.render(&prompt, &[]);
        let intro_idx = text.find("## Core Principles").unwrap();
        let guidance_idx = text.find("## Guidance").unwrap();
        let env_idx = text.find("cwd: /repo").unwrap();
        let tools_idx = text.find("## Available Tools").unwrap();
        assert!(intro_idx < guidance_idx);
        assert!(guidance_idx < env_idx);
        assert!(env_idx < tools_idx);
    }

    #[test]
    fn prompt_does_not_emit_runtime_prune_status() {
        let text = DefaultPromptRenderer.render(&prompt(crate::ExecutionMode::Repl), &[]);
        assert!(!text.contains("Context-pruned turns this run"));
        assert!(!text.contains("Skip history-mining detours"));
    }
}
