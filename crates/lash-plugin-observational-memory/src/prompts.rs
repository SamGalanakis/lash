use lash_core::{MessageRole, PartKind};

use crate::constants::{
    OBSERVER_EXTRACTION_INSTRUCTIONS, OBSERVER_GUIDELINES, OBSERVER_OUTPUT_FORMAT_BASE,
    REFLECTOR_SYSTEM_PROMPT_PREFIX, REFLECTOR_SYSTEM_PROMPT_SUFFIX,
};
use crate::model::{ObservedMessageNode, ParsedMemoryOutput};

pub(crate) fn truncate_observation_tail(observations: &str, budget_tokens: usize) -> String {
    if budget_tokens == 0 {
        return String::new();
    }
    let budget_chars = budget_tokens.saturating_mul(4);
    let chars = observations.chars().collect::<Vec<_>>();
    if chars.len() <= budget_chars {
        return observations.to_string();
    }
    let start = chars.len().saturating_sub(budget_chars);
    let tail = chars[start..].iter().collect::<String>();
    format!("…\n{tail}")
}

pub(crate) fn build_observer_prompt(
    existing_observations: Option<&str>,
    messages: &[impl ObservedMessageNode],
    prior_current_task: Option<&str>,
    prior_suggested_response: Option<&str>,
) -> String {
    let formatted_messages = messages
        .iter()
        .map(format_message_for_observer)
        .collect::<Vec<_>>()
        .join("\n\n");

    let mut prompt =
        format!("## New Message History to Observe\n\n{formatted_messages}\n\n---\n\n");
    if let Some(existing) = existing_observations.filter(|text| !text.trim().is_empty()) {
        prompt.push_str("## Previous Observations\n\n");
        prompt.push_str(existing.trim());
        prompt.push_str("\n\n---\n\nDo not repeat these existing observations. Your new observations will be appended to the existing observations.\n\n");
    }
    let mut prior_lines = Vec::new();
    if let Some(current_task) = prior_current_task.filter(|text| !text.trim().is_empty()) {
        prior_lines.push(format!("- prior current-task: {}", current_task.trim()));
    }
    if let Some(suggested_response) =
        prior_suggested_response.filter(|text| !text.trim().is_empty())
    {
        prior_lines.push(format!(
            "- prior suggested-response: {}",
            suggested_response.trim()
        ));
    }
    if !prior_lines.is_empty() {
        prompt.push_str("## Existing Continuity Signals\n");
        prompt.push_str(&prior_lines.join("\n"));
        prompt.push_str("\n\n");
    }
    prompt.push_str("## Task\nObserve only the NEW message history above and produce incremental memory updates.\n");
    prompt
}

pub(crate) fn build_reflector_prompt(observations: &str) -> String {
    format!(
        "## Observations to Reflect\n\n{}\n\n## Task\nRestructure and compress these observations while preserving the most important user facts, active tasks, and continuity cues.",
        observations.trim()
    )
}

pub(crate) fn observer_system_prompt() -> String {
    format!(
        "{}\n\n{}\n\n{}\n\n{}",
        OBSERVER_EXTRACTION_INSTRUCTIONS,
        OBSERVER_OUTPUT_FORMAT_BASE,
        OBSERVER_GUIDELINES,
        "Output only the XML blocks."
    )
}

pub(crate) fn reflector_system_prompt() -> String {
    format!(
        "{}\n{}\n\n{}",
        REFLECTOR_SYSTEM_PROMPT_PREFIX,
        observer_system_prompt(),
        REFLECTOR_SYSTEM_PROMPT_SUFFIX
    )
}

pub(crate) fn parse_memory_output(text: &str) -> ParsedMemoryOutput {
    let observations =
        capture_xml_block(text, "observations").unwrap_or_else(|| text.trim().to_string());
    let current_task =
        capture_xml_block(text, "current-task").filter(|value| !value.trim().is_empty());
    let suggested_response =
        capture_xml_block(text, "suggested-response").filter(|value| !value.trim().is_empty());
    ParsedMemoryOutput {
        observations: observations.trim().to_string(),
        current_task: current_task.map(|value| value.trim().to_string()),
        suggested_response: suggested_response.map(|value| value.trim().to_string()),
    }
}

pub(crate) fn capture_xml_block(text: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = text.find(&open)? + open.len();
    let end = text[start..].find(&close)?;
    Some(text[start..start + end].trim().to_string())
}

pub(crate) fn format_message_for_observer(node: &impl ObservedMessageNode) -> String {
    let timestamp = format_observation_timestamp(node.timestamp());
    let role = match node.message().role {
        MessageRole::User => "USER",
        MessageRole::Assistant => "ASSISTANT",
        MessageRole::System => "SYSTEM",
        MessageRole::Event => "EVENT",
    };
    let content = node
        .message()
        .parts
        .iter()
        .map(|part| match part.kind {
            PartKind::Text | PartKind::Prose | PartKind::Output | PartKind::Error => {
                part.content.clone()
            }
            PartKind::Code => format!("```{}\n```", part.content),
            PartKind::ToolCall => format!("[tool call] {}", part.content),
            PartKind::ToolResult => format!("[tool result] {}", part.content),
            PartKind::Image => "[image]".to_string(),
            // Reasoning parts are excluded from observational memory —
            // chain-of-thought is display-only and not durable context.
            PartKind::Reasoning => String::new(),
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("[{timestamp}] {role}\n{}", content.trim())
}

pub(crate) fn format_observation_timestamp(timestamp: &str) -> String {
    chrono::DateTime::parse_from_rfc3339(timestamp)
        .map(|dt| dt.format("%b %-d, %Y %H:%M").to_string())
        .unwrap_or_else(|_| timestamp.to_string())
}
