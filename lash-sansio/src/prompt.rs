use std::collections::BTreeSet;

use crate::{ExecutionMode, PromptContext, PromptContribution, PromptTemplate};

#[derive(Clone, Debug)]
pub struct PromptBuildInput {
    pub mode: ExecutionMode,
    pub template: PromptTemplate,
    pub execution_prompt: String,
    pub tool_names: Vec<String>,
    pub omitted_tool_count: usize,
    pub contributions: Vec<PromptContribution>,
}

#[derive(Clone, Debug)]
pub struct PreparedPrompt {
    pub context: PromptContext,
    pub system_prompt: String,
}

pub fn build_prompt(input: PromptBuildInput) -> PreparedPrompt {
    let context = PromptContext {
        mode: input.mode,
        execution_prompt: input.execution_prompt,
        tool_names: input.tool_names,
        omitted_tool_count: input.omitted_tool_count,
        contributions: merge_prompt_contributions(input.contributions),
    };
    let system_prompt = input.template.render(&context);
    PreparedPrompt {
        context,
        system_prompt,
    }
}

fn merge_prompt_contributions(contributions: Vec<PromptContribution>) -> Vec<PromptContribution> {
    let mut merged = contributions
        .into_iter()
        .filter_map(normalize_contribution)
        .collect::<Vec<_>>();

    merged.sort_by(|left, right| {
        slot_order(left.slot)
            .cmp(&slot_order(right.slot))
            .then(left.priority.cmp(&right.priority))
            .then_with(|| left.title.cmp(&right.title))
            .then_with(|| left.content.cmp(&right.content))
    });

    let mut seen = BTreeSet::new();
    merged.retain(|contribution| {
        seen.insert((
            slot_order(contribution.slot),
            contribution.priority,
            contribution.title.clone().unwrap_or_default(),
            contribution.content.clone(),
        ))
    });
    merged
}

fn normalize_contribution(mut contribution: PromptContribution) -> Option<PromptContribution> {
    contribution.content = contribution.content.trim().to_string();
    if contribution.content.is_empty() {
        return None;
    }
    contribution.title = contribution
        .title
        .as_deref()
        .map(str::trim)
        .filter(|title| !title.is_empty())
        .map(ToOwned::to_owned);
    Some(contribution)
}

fn slot_order(slot: crate::PromptSlot) -> usize {
    match slot {
        crate::PromptSlot::Intro => 0,
        crate::PromptSlot::Execution => 1,
        crate::PromptSlot::Guidance => 2,
        crate::PromptSlot::ProjectInstructions => 3,
        crate::PromptSlot::RuntimeContext => 4,
        crate::PromptSlot::Environment => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PromptContribution, default_prompt_template};

    #[test]
    fn build_prompt_renders_template_from_merged_context() {
        let prepared = build_prompt(PromptBuildInput {
            mode: crate::ExecutionMode::Standard,
            template: default_prompt_template(),
            execution_prompt: "Use tools.".to_string(),
            tool_names: vec!["read_file".to_string()],
            omitted_tool_count: 0,
            contributions: vec![
                PromptContribution::guidance("Repo", "Follow repo rules."),
                PromptContribution::guidance("Repo", "Follow repo rules."),
                PromptContribution::project_instructions("Be careful."),
            ],
        });

        assert!(prepared.system_prompt.contains("Use tools."));
        assert!(prepared.system_prompt.contains("Follow repo rules."));
        assert!(prepared.system_prompt.contains("Be careful."));
        assert_eq!(prepared.context.contributions.len(), 2);
    }
}
