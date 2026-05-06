use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

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
    pub system_prompt: Arc<str>,
}

/// Single-slot memo for the rendered system prompt, keyed by a hash of
/// the inputs. Most consecutive turns in a session pass identical
/// inputs (template, mode-preamble-derived `execution_prompt` /
/// `tool_names`, plus context contributions), so a one-slot cache hits
/// repeatedly and avoids the section-by-section `Vec<String>::join`
/// work in `PromptTemplate::render`.
#[derive(Default)]
pub struct PromptCache {
    inner: Mutex<Option<(u64, Arc<str>)>>,
}

impl PromptCache {
    pub fn new() -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for PromptCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromptCache").finish_non_exhaustive()
    }
}

pub fn build_prompt(input: PromptBuildInput) -> PreparedPrompt {
    build_prompt_cached(input, None)
}

pub fn build_prompt_cached(input: PromptBuildInput, cache: Option<&PromptCache>) -> PreparedPrompt {
    let context = PromptContext {
        mode: input.mode,
        execution_prompt: input.execution_prompt,
        tool_names: input.tool_names,
        omitted_tool_count: input.omitted_tool_count,
        contributions: merge_prompt_contributions(input.contributions),
    };
    let key = cache.map(|_| hash_prompt_inputs(&input.template, &context));
    if let (Some(cache), Some(key)) = (cache, key)
        && let Some(cached) = cache.inner.lock().ok().and_then(|guard| {
            guard
                .as_ref()
                .filter(|(k, _)| *k == key)
                .map(|(_, v)| Arc::clone(v))
        })
    {
        return PreparedPrompt {
            context,
            system_prompt: cached,
        };
    }
    let system_prompt: Arc<str> = Arc::from(input.template.render(&context));
    if let (Some(cache), Some(key)) = (cache, key)
        && let Ok(mut guard) = cache.inner.lock()
    {
        *guard = Some((key, Arc::clone(&system_prompt)));
    }
    PreparedPrompt {
        context,
        system_prompt,
    }
}

fn hash_prompt_inputs(template: &PromptTemplate, context: &PromptContext) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    template.hash(&mut hasher);
    context.mode.hash(&mut hasher);
    context.execution_prompt.hash(&mut hasher);
    context.tool_names.hash(&mut hasher);
    context.omitted_tool_count.hash(&mut hasher);
    for contribution in &context.contributions {
        contribution.slot.hash(&mut hasher);
        contribution.priority.hash(&mut hasher);
        contribution.title.hash(&mut hasher);
        contribution.content.hash(&mut hasher);
    }
    hasher.finish()
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

    // Duplicates are adjacent after the sort, so `dedup_by` on &str
    // refs drops them without cloning anything.
    merged.dedup_by(|a, b| {
        slot_order(a.slot) == slot_order(b.slot)
            && a.priority == b.priority
            && a.title.as_deref() == b.title.as_deref()
            && a.content == b.content
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
        crate::PromptSlot::CliAutonomousIntro => 2,
        crate::PromptSlot::CliAutonomousExecution => 3,
        crate::PromptSlot::CliRlmExecution => 4,
        crate::PromptSlot::Guidance => 5,
        crate::PromptSlot::ProjectInstructions => 6,
        crate::PromptSlot::RuntimeContext => 7,
        crate::PromptSlot::Environment => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PromptContribution, default_prompt_template};

    #[test]
    fn build_prompt_renders_template_from_merged_context() {
        let prepared = build_prompt(PromptBuildInput {
            mode: crate::ExecutionMode::standard(),
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

    #[test]
    fn build_prompt_cached_reuses_arc_on_identical_inputs() {
        let cache = PromptCache::new();
        let inputs = || PromptBuildInput {
            mode: crate::ExecutionMode::standard(),
            template: default_prompt_template(),
            execution_prompt: "Use tools.".to_string(),
            tool_names: vec!["read_file".to_string()],
            omitted_tool_count: 0,
            contributions: vec![PromptContribution::guidance("Repo", "Follow repo rules.")],
        };
        let first = build_prompt_cached(inputs(), Some(&cache));
        let second = build_prompt_cached(inputs(), Some(&cache));
        assert!(Arc::ptr_eq(&first.system_prompt, &second.system_prompt));
    }

    #[test]
    fn build_prompt_cached_renders_again_when_inputs_change() {
        let cache = PromptCache::new();
        let first = build_prompt_cached(
            PromptBuildInput {
                mode: crate::ExecutionMode::standard(),
                template: default_prompt_template(),
                execution_prompt: "Use tools.".to_string(),
                tool_names: vec!["read_file".to_string()],
                omitted_tool_count: 0,
                contributions: vec![],
            },
            Some(&cache),
        );
        let second = build_prompt_cached(
            PromptBuildInput {
                mode: crate::ExecutionMode::standard(),
                template: default_prompt_template(),
                execution_prompt: "Use other tools.".to_string(),
                tool_names: vec!["read_file".to_string()],
                omitted_tool_count: 0,
                contributions: vec![],
            },
            Some(&cache),
        );
        assert!(!Arc::ptr_eq(&first.system_prompt, &second.system_prompt));
        assert_ne!(first.system_prompt, second.system_prompt);
    }
}
