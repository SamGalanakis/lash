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
}

impl PromptSectionName {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Intro => "intro",
            Self::Execution => "execution",
            Self::Guidance => "guidance",
            Self::Environment => "environment",
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block: Option<String>,
    pub mode: PromptOverrideMode,
    #[serde(default)]
    pub content: String,
}

pub trait PromptRenderer: Send + Sync {
    fn render(&self, prompt: &PromptContext, overrides: &[PromptSectionOverride]) -> String;
}

type PromptSectionBuilder = for<'a> fn(&PromptRenderContext<'a>) -> Option<String>;
type PromptBlockBuilder = for<'a> fn(&PromptRenderContext<'a>) -> Vec<PromptBlockState>;

struct PromptSectionDef {
    name: PromptSectionName,
    title: Option<&'static str>,
    builder: PromptSectionBuilder,
    block_builder: PromptBlockBuilder,
}

#[derive(Clone, Debug, Default)]
struct PromptSectionState {
    body: Option<String>,
    blocks: Vec<PromptBlockState>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PromptBlockState {
    key: String,
    title: Option<String>,
    content: String,
}

impl PromptBlockState {
    fn new(key: impl Into<String>, title: Option<String>, content: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            title,
            content: content.into(),
        }
    }
}

const SECTION_DEFS: [PromptSectionDef; 4] = [
    PromptSectionDef {
        name: PromptSectionName::Intro,
        title: None,
        builder: intro_section,
        block_builder: no_blocks,
    },
    PromptSectionDef {
        name: PromptSectionName::Execution,
        title: Some("Execution"),
        builder: execution_section,
        block_builder: execution_blocks,
    },
    PromptSectionDef {
        name: PromptSectionName::Guidance,
        title: Some("Guidance"),
        builder: guidance_section,
        block_builder: no_blocks,
    },
    PromptSectionDef {
        name: PromptSectionName::Environment,
        title: Some("Environment"),
        builder: environment_section,
        block_builder: no_blocks,
    },
];

struct PromptRenderContext<'a> {
    prompt: &'a PromptContext,
}

impl<'a> PromptRenderContext<'a> {
    fn new(prompt: &'a PromptContext) -> Self {
        Self { prompt }
    }
}

#[derive(Default)]
pub struct DefaultPromptRenderer;

impl PromptRenderer for DefaultPromptRenderer {
    fn render(&self, prompt: &PromptContext, overrides: &[PromptSectionOverride]) -> String {
        let render_context = PromptRenderContext::new(prompt);
        let mut sections: HashMap<PromptSectionName, PromptSectionState> = HashMap::new();
        for def in SECTION_DEFS {
            sections.insert(
                def.name,
                PromptSectionState {
                    body: (def.builder)(&render_context),
                    blocks: (def.block_builder)(&render_context),
                },
            );
        }
        apply_prompt_contributions(&mut sections, &prompt.contributions);
        apply_block_overrides(&mut sections, overrides);

        let mut rendered_sections: HashMap<PromptSectionName, Option<String>> = HashMap::new();
        for def in SECTION_DEFS {
            let rendered = sections
                .get(&def.name)
                .and_then(|state| render_section(def.title, state));
            rendered_sections.insert(def.name, rendered);
        }
        apply_section_overrides(&mut rendered_sections, overrides);

        SECTION_DEFS
            .iter()
            .filter_map(|def| rendered_sections.get(&def.name).cloned().flatten())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

pub fn default_prompt_renderer() -> Arc<dyn PromptRenderer> {
    Arc::new(DefaultPromptRenderer)
}

fn no_blocks(_ctx: &PromptRenderContext<'_>) -> Vec<PromptBlockState> {
    Vec::new()
}

fn execution_blocks(ctx: &PromptRenderContext<'_>) -> Vec<PromptBlockState> {
    if matches!(ctx.prompt.mode, crate::ExecutionMode::Repl)
        && !ctx.prompt.tool_list.trim().is_empty()
    {
        vec![PromptBlockState::new(
            "available_tools",
            Some("Available Tools".to_string()),
            ctx.prompt.tool_list.trim().to_string(),
        )]
    } else {
        Vec::new()
    }
}

fn apply_section_overrides(
    sections: &mut HashMap<PromptSectionName, Option<String>>,
    overrides: &[PromptSectionOverride],
) {
    for ov in overrides.iter().filter(|ov| ov.block.is_none()) {
        let entry = sections.entry(ov.section).or_insert(None);
        *entry = apply_text_override(entry.take(), ov.mode, &ov.content);
    }
}

fn apply_prompt_contributions(
    sections: &mut HashMap<PromptSectionName, PromptSectionState>,
    contributions: &[PromptContribution],
) {
    let mut ordered = contributions.to_vec();
    ordered.sort_by_key(|contribution| contribution.priority);
    for contribution in &ordered {
        let content = contribution.content.trim();
        if content.is_empty() {
            continue;
        }
        let entry = sections.entry(contribution.section).or_default();
        upsert_block(
            &mut entry.blocks,
            contribution.block.clone(),
            contribution.title.clone(),
            content.to_string(),
        );
    }
}

fn apply_block_overrides(
    sections: &mut HashMap<PromptSectionName, PromptSectionState>,
    overrides: &[PromptSectionOverride],
) {
    for ov in overrides.iter().filter(|ov| ov.block.is_some()) {
        let state = sections.entry(ov.section).or_default();
        apply_block_override(state, ov);
    }
}

fn apply_block_override(state: &mut PromptSectionState, override_def: &PromptSectionOverride) {
    let Some(block_key) = override_def.block.as_deref() else {
        return;
    };
    let Some(index) = state.blocks.iter().position(|block| block.key == block_key) else {
        if matches!(override_def.mode, PromptOverrideMode::Disable) {
            return;
        }
        let title = Some(titleize_block_key(block_key));
        let block = PromptBlockState::new(block_key, title, override_def.content.clone());
        if matches!(override_def.mode, PromptOverrideMode::Prepend) {
            state.blocks.insert(0, block);
        } else {
            state.blocks.push(block);
        }
        return;
    };

    let existing = state.blocks[index].content.clone();
    match apply_text_override(Some(existing), override_def.mode, &override_def.content) {
        Some(content) => state.blocks[index].content = content,
        None => {
            state.blocks.remove(index);
        }
    }
}

fn apply_text_override(
    existing: Option<String>,
    mode: PromptOverrideMode,
    content: &str,
) -> Option<String> {
    match mode {
        PromptOverrideMode::Disable => None,
        PromptOverrideMode::Replace => Some(content.to_string()),
        PromptOverrideMode::Prepend => {
            let existing = existing.unwrap_or_default();
            Some(if existing.trim().is_empty() {
                content.to_string()
            } else {
                format!("{content}\n\n{existing}")
            })
        }
        PromptOverrideMode::Append => {
            let existing = existing.unwrap_or_default();
            Some(if existing.trim().is_empty() {
                content.to_string()
            } else {
                format!("{existing}\n\n{content}")
            })
        }
    }
}

fn upsert_block(
    blocks: &mut Vec<PromptBlockState>,
    key: String,
    title: Option<String>,
    content: String,
) {
    if let Some(existing) = blocks.iter_mut().find(|block| block.key == key) {
        if existing.title.is_none() {
            existing.title = title;
        }
        if existing.content.trim().is_empty() {
            existing.content = content;
        } else {
            existing.content.push_str("\n\n");
            existing.content.push_str(content.trim());
        }
        return;
    }

    blocks.push(PromptBlockState::new(key, title, content));
}

fn render_section(title: Option<&'static str>, state: &PromptSectionState) -> Option<String> {
    let body = state
        .body
        .as_deref()
        .map(str::trim)
        .filter(|body| !body.is_empty())
        .map(str::to_string);
    let blocks = state
        .blocks
        .iter()
        .filter_map(render_block)
        .collect::<Vec<_>>();

    if body.is_none() && blocks.is_empty() {
        return None;
    }

    let mut parts = Vec::new();
    if let Some(title) = title {
        parts.push(format!("## {title}"));
    }
    if let Some(body) = body {
        parts.push(body);
    }
    parts.extend(blocks);
    Some(parts.join("\n\n"))
}

fn render_block(block: &PromptBlockState) -> Option<String> {
    let content = block.content.trim();
    if content.is_empty() {
        return None;
    }
    match block
        .title
        .as_deref()
        .map(str::trim)
        .filter(|title| !title.is_empty())
    {
        Some(title) => Some(format!("### {title}\n\n{content}")),
        None => Some(content.to_string()),
    }
}

fn titleize_block_key(key: &str) -> String {
    key.split(|ch: char| ch == '_' || ch == '-' || ch == '.')
        .filter(|segment| !segment.is_empty())
        .map(|segment| {
            let mut chars = segment.chars();
            match chars.next() {
                Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

const MAIN_AGENT_INTRO: &str = "You are an AI coding assistant inside lash.";

const REPL_EXECUTION_SECTION: &str = "Your output can include prose and `<repl>` blocks.\n- Work iteratively: inspect, act, observe, continue\n- Most tasks take multiple REPL cycles, not one large block\n- Use at most one `<repl>` block per response; once you close `</repl>`, stop and wait for the result\n- If you need tools or execution, emit a `<repl>` block and stop there\n- If the task is complete, do not emit `<repl>`; reply in plain prose and that finalizes the turn\n- Never put user-facing prose after `</repl>`; anything after the first closed block will be ignored\n- Use `observe` for intermediate results, inspection, and progress that should continue; `observe` output is hidden from the user\n- Verify the concrete end state before replying in prose when possible\n\n### REPL Language\n\nThe REPL is `lashlang`, a small workflow language for tool orchestration.\n- Values are null, booleans, numbers, strings, lists, and records\n- List and record literals use comma-separated entries: `[a, b]`, `{ a: 1, b: 2 }`; tool arg records follow the same rule\n- Assign with `name = expr`\n- Bare expressions are valid statements; in `parallel { ... }`, a bare expression branch contributes that value to the result list\n- Call tools with `call tool_name { arg: expr }`\n- Use `parallel { ... }` only for independent tool calls; if one call needs another call's output, do not put them in the same `parallel { ... }`\n- `parallel { ... }` returns a list of branch results in source order, and branches that end with `call ...` produce the same wrapped `{ ok, value, error }` records as ordinary tool calls\n- Use `observe expr` to inspect a value and continue execution\n- Control flow is limited to statement `if` and `for`; `parallel` also works as an expression\n- Use ternary expressions for inline branching: `cond ? yes : no`\n- Boolean negation supports both `!cond` and `not cond`\n- Boolean conjunction/disjunction support both `&&` / `||` and `and` / `or`\n- Tool results are records like `{ ok: true, value: ... }` or `{ ok: false, error: ... }`\n- Access the wrapped payload via `.value` only when `result.ok` is true\n- Do not assume every `value` is a record: many tools return strings, numbers, or lists directly\n- Builtins: `len`, `empty`, `contains`, `slice`, `json_parse`, `format`, `to_string`, `to_int`, `to_float`\n- Builtins return plain values; invalid builtin usage raises a runtime error instead of returning a `{ ok, error }` record\n- `slice(value, start, end)` treats `null` bounds as omitted: `start=null` means from the beginning, `end=null` means through the end\n- `to_string(value)` stringifies a single value\n- `format(\"...\", args...)` formats templates with `{}` placeholders; use `{0}`, `{1}`, ... only when argument reordering matters\n- Escape literal braces in templates with `{{` and `}}`\n- String `+` concatenation auto-stringifies when either side is already a string";

const STANDARD_EXECUTION_SECTION: &str = "Use direct tool calls when execution is needed.\n- Do not emit `<repl>` blocks or Python code\n- Call tools directly with valid arguments\n- Use `batch` for 2 or more independent tool calls; serialize only when later arguments depend on earlier results\n- Avoid filler prose between tool calls\n- Work in small, concrete steps and verify each meaningful step before broadening scope\n- After edits, run the narrowest check that can falsify the change before moving on to broader validation\n- If a tool fails or returns incomplete output, inspect the current state, fix the cause, and continue; do not repeat the same failing call unchanged\n- Keep going until the task is complete; do not stop after inspection or partial progress\n- If you are unsure, resolve the uncertainty with the smallest relevant check; broaden only when the current path is insufficient\n- Before concluding, verify the concrete end-state with tools whenever possible\n- For direct conversational requests that need no tools, respond in prose only\n- Finish by returning a final assistant answer once the task is actually complete";

fn intro_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    let _ = ctx;
    Some(MAIN_AGENT_INTRO.to_string())
}

fn execution_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    Some(if matches!(ctx.prompt.mode, crate::ExecutionMode::Repl) {
        REPL_EXECUTION_SECTION.to_string()
    } else {
        STANDARD_EXECUTION_SECTION.to_string()
    })
}

fn guidance_section(ctx: &PromptRenderContext<'_>) -> Option<String> {
    let _ = ctx;
    None
}

fn environment_section(_ctx: &PromptRenderContext<'_>) -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prompt(mode: crate::ExecutionMode) -> PromptContext {
        PromptContext {
            mode,
            tool_list: "tools".to_string(),
            ..PromptContext::default()
        }
    }

    #[test]
    fn parses_prompt_section_names() {
        assert_eq!(
            PromptSectionName::from_str("guidance").unwrap(),
            PromptSectionName::Guidance
        );
        assert!(PromptSectionName::from_str("available_tools").is_err());
        assert!(PromptSectionName::from_str("memory_api").is_err());
    }

    #[test]
    fn overrides_apply_in_order() {
        let overrides = vec![
            PromptSectionOverride {
                section: PromptSectionName::Intro,
                block: None,
                mode: PromptOverrideMode::Replace,
                content: "A".into(),
            },
            PromptSectionOverride {
                section: PromptSectionName::Intro,
                block: None,
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
        assert!(text.contains("### Available Tools"));
    }

    #[test]
    fn standard_prompt_keeps_only_execution_contract_by_default() {
        let text = DefaultPromptRenderer.render(&prompt(crate::ExecutionMode::Standard), &[]);
        assert!(text.contains("Work in small, concrete steps and verify each meaningful step"));
        assert!(text.contains("Before concluding, verify the concrete end-state"));
        assert!(text.contains("resolve the uncertainty with the smallest relevant check"));
        assert!(!text.contains("Default to concise, direct, friendly communication"));
        assert!(!text.contains("Prefer short final answers; use a compact paragraph or bullets"));
        assert!(!text.contains("update_plan"));
        assert!(!text.contains("### Available Tools"));
    }

    #[test]
    fn guidance_contributions_render_without_default_guidance_wrapper() {
        let mut prompt = prompt(crate::ExecutionMode::Repl);
        prompt.contributions = vec![
            PromptContribution::guidance("first_guide", "First Guide", "First details."),
            PromptContribution::guidance("second_guide", "Second Guide", "Second details."),
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
            PromptContribution::environment("runtime_context", "Runtime Context", "cwd: /tmp/demo"),
            PromptContribution::guidance(
                "project_instructions",
                "Project Instructions",
                "Do the safe thing.",
            ),
        ];
        let text = DefaultPromptRenderer.render(&prompt, &[]);
        assert!(text.contains("## Environment"));
        assert!(text.contains("### Runtime Context"));
        assert!(text.contains("cwd: /tmp/demo"));
        assert!(text.contains("### Project Instructions"));
    }

    #[test]
    fn default_prompt_no_longer_emits_shared_guidance_block() {
        let repl = DefaultPromptRenderer.render(&prompt(crate::ExecutionMode::Repl), &[]);
        let standard = DefaultPromptRenderer.render(&prompt(crate::ExecutionMode::Standard), &[]);

        assert!(!repl.contains("## Guidance"));
        assert!(!standard.contains("## Guidance"));
        assert!(!repl.contains("Default to concise, direct, friendly communication"));
        assert!(!standard.contains("Default to concise, direct, friendly communication"));
    }

    #[test]
    fn prompt_orders_environment_after_execution() {
        let mut prompt = prompt(crate::ExecutionMode::Repl);
        prompt.contributions = vec![
            PromptContribution::guidance("custom", "Custom", "More guidance."),
            PromptContribution::environment("runtime_context", "Runtime Context", "cwd: /repo"),
        ];
        let text = DefaultPromptRenderer.render(&prompt, &[]);
        let intro_idx = text.find(MAIN_AGENT_INTRO).unwrap();
        let guidance_idx = text.find("## Guidance").unwrap();
        let env_idx = text.find("## Environment").unwrap();
        assert!(intro_idx < guidance_idx);
        assert!(guidance_idx < env_idx);
        assert!(text.contains("### Available Tools"));
    }

    #[test]
    fn block_overrides_can_target_structured_prompt_blocks() {
        let mut prompt = prompt(crate::ExecutionMode::Repl);
        prompt.contributions = vec![PromptContribution::guidance(
            "project_instructions",
            "Project Instructions",
            "Follow the repo rules.",
        )];
        let overrides = vec![PromptSectionOverride {
            section: PromptSectionName::Guidance,
            block: Some("project_instructions".to_string()),
            mode: PromptOverrideMode::Replace,
            content: "Use the local conventions.".to_string(),
        }];
        let text = DefaultPromptRenderer.render(&prompt, &overrides);
        assert!(text.contains("### Project Instructions"));
        assert!(text.contains("Use the local conventions."));
        assert!(!text.contains("Follow the repo rules."));
    }

    #[test]
    fn prompt_does_not_emit_runtime_prune_status() {
        let text = DefaultPromptRenderer.render(&prompt(crate::ExecutionMode::Repl), &[]);
        assert!(!text.contains("Context-pruned turns this run"));
        assert!(!text.contains("Skip history-mining detours"));
    }
}
