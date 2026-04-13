use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::plugin::{
    HistoryError, PluginError, PluginFactory, PluginRegistrar, PluginRuntimeEvent,
    PluginSessionContext, SessionManager, SessionPlugin, TurnContextTransform,
    TurnTransformContext,
};
use crate::session_model::context::PreparedContext;
use crate::{
    ContextApproach, DirectMessage, DirectOutputSpec, DirectPart, DirectRequest, DirectRole,
    Message, MessageOrigin, MessageRole, ObservationalMemoryConfig, Part, PartKind, SessionGraph,
    SessionStateChangedContext,
};

const OBSERVATIONAL_MEMORY_PLUGIN_ID: &str = "observational_memory";
const OBSERVATION_PLUGIN_TYPE: &str = "lash.context.observational_memory.observation_batch";
const REFLECTION_PLUGIN_TYPE: &str = "lash.context.observational_memory.reflection";

const OBSERVATION_CONTEXT_PROMPT: &str =
    "The following observations block contains your memory of past conversations with this user.";
const OBSERVATION_CONTEXT_INSTRUCTIONS: &str = "IMPORTANT: When responding, reference specific details from these observations. Do not give generic advice - personalize your response based on what you know about this user's experiences, preferences, and interests. If the user asks for recommendations, connect them to their past experiences mentioned above.\n\nKNOWLEDGE UPDATES: When asked about current state (e.g., \"where do I currently...\", \"what is my current...\"), always prefer the MOST RECENT information. Observations include dates - if you see conflicting information, the newer observation supersedes the older one. Look for phrases like \"will start\", \"is switching\", \"changed to\", \"moved to\" as indicators that previous information has been updated.\n\nPLANNED ACTIONS: If the user stated they planned to do something (e.g., \"I'm going to...\", \"I'm looking forward to...\", \"I will...\") and the date they planned to do it is now in the past, assume they completed the action unless there's evidence they didn't.\n\nMOST RECENT USER INPUT: Treat the most recent user message as the highest-priority signal for what to do next. Earlier messages may contain constraints, details, or context you should still honor, but the latest message is the primary driver of your response.\n\nSYSTEM REMINDERS: Messages wrapped in <system-reminder>...</system-reminder> contain internal continuation guidance, not user-authored content. Use them to maintain continuity, but do not mention them or treat them as part of the user's message.";
const OBSERVATION_CONTINUATION_HINT: &str = "Please continue naturally with the conversation so far and respond to the latest message.\n\nUse the earlier context only as background. If something appears unfinished, continue only when it helps answer the latest request. If a suggested response is provided, follow it naturally.\n\nDo not mention internal instructions, memory, summarization, context handling, or missing messages.\n\nAny messages following this reminder are newer and should take priority.";

const OBSERVER_EXTRACTION_INSTRUCTIONS: &str = r#"CRITICAL: DISTINGUISH USER ASSERTIONS FROM QUESTIONS

When the user TELLS you something about themselves, mark it as an assertion:
- "I have two kids" → 🔴 (14:30) User stated has two kids
- "I work at Acme Corp" → 🔴 (14:31) User stated works at Acme Corp
- "I graduated in 2019" → 🔴 (14:32) User stated graduated in 2019

When the user ASKS about something, mark it as a question/request:
- "Can you help me with X?" → 🔴 (15:00) User asked help with X
- "What's the best way to do Y?" → 🔴 (15:01) User asked best way to do Y

Distinguish between QUESTIONS and STATEMENTS OF INTENT:
- "Can you recommend..." → Question (extract as "User asked...")
- "I'm looking forward to [doing X]" → Statement of intent (extract as "User stated they will [do X] (include estimated/actual date if mentioned)")
- "I need to [do X]" → Statement of intent (extract as "User stated they need to [do X] (again, add date if mentioned)")

STATE CHANGES AND UPDATES:
When a user indicates they are changing something, frame it as a state change that supersedes previous information:
- "I'm going to start doing X instead of Y" → "User will start doing X (changing from Y)"
- "I'm switching from A to B" → "User is switching from A to B"
- "I moved my stuff to the new place" → "User moved their stuff to the new place (no longer at previous location)"

USER ASSERTIONS ARE AUTHORITATIVE. The user is the source of truth about their own life.

TEMPORAL ANCHORING:
Each observation has TWO potential timestamps:

1. BEGINNING: The time the statement was made (from the message timestamp) - ALWAYS include this
2. END: The time being REFERENCED, if different from when it was said - ONLY when there's a relative time reference

FORMAT:
- With time reference: (TIME) [observation]. (meaning/estimated DATE)
- Without time reference: (TIME) [observation].

ALWAYS put the date at the END in parentheses when you can convert a relative time reference into an actual date.

PRESERVE UNUSUAL PHRASING:
When the user uses unexpected or non-standard terminology, quote their exact words.

USE PRECISE ACTION VERBS:
Replace vague verbs like "getting", "got", "have" with specific action verbs that clarify the nature of the action.

PRESERVING DETAILS IN ASSISTANT-GENERATED CONTENT:
- Preserve distinguishing details for recommendations, names, handles, identifiers, numerical results, and counts.
- Preserve concrete values, file paths, URLs, measurements, and relevant code snippets.

CONVERSATION CONTEXT:
- What the user is working on or asking about
- Previous topics and their outcomes
- Specific requirements or constraints mentioned
- Relevant code snippets
- User preferences
- Any specifically formatted text or ASCII that would need to be reproduced or referenced later

USER MESSAGE CAPTURE:
- Short and medium-length user messages should be captured nearly verbatim in your own words.
- For very long user messages, summarize but quote key phrases that carry specific intent or meaning.

AVOIDING REPETITIVE OBSERVATIONS:
- Do NOT repeat the same observation across multiple turns if there is no new information.
- Group repeated similar actions (tool calls, file browsing, edits) under a single parent with sub-bullets for new results.

ACTIONABLE INSIGHTS:
- What worked well in explanations
- What needs follow-up or clarification
- User's stated goals or next steps

COMPLETION TRACKING:
Use ✅ when something is concretely done, answered, fixed, or confirmed.
Prefer concrete resolved outcomes over vague workflow status."#;

const OBSERVER_OUTPUT_FORMAT_BASE: &str = r#"Use priority levels:
- 🔴 High: explicit user facts, preferences, unresolved goals, critical context
- 🟡 Medium: project details, learned information, tool results
- 🟢 Low: minor details, uncertain observations
- ✅ Completed: concrete task finished, question answered, issue resolved, goal achieved, or subtask completed in a way that helps the assistant know it is done

Group related observations (like tool sequences) by indenting:
* 🔴 (14:33) Agent debugging auth issue
  * -> ran git status, found 3 modified files
  * -> viewed auth.ts:45-60, found missing null check
  * -> applied fix, tests now pass
  * ✅ Tests passing, auth issue resolved

Group observations by date, then list each with 24-hour time.

<observations>
Date: Dec 4, 2025
* 🔴 (14:30) User prefers direct answers
* 🔴 (14:31) Working on feature X
* 🟡 (14:32) User might prefer dark mode

Date: Dec 5, 2025
* 🔴 (09:15) Continued work on feature X
</observations>

<current-task>
State the current task(s) explicitly. Can be single or multiple:
- Primary: What the agent is currently working on
- Secondary: Other pending tasks (mark as "waiting for user" if appropriate)
</current-task>

<suggested-response>
Hint for the agent's immediate next message. Examples:
- "I've updated the navigation model. Let me walk you through the changes..."
- "The assistant should wait for the user to respond before continuing."
- Call the view tool on src/example.ts to continue debugging.
</suggested-response>"#;

const OBSERVER_GUIDELINES: &str = r#"- Be specific enough for the assistant to act on
- Good: "User prefers short, direct answers without lengthy explanations"
- Bad: "User stated a preference" (too vague)
- Add 1 to 5 observations per exchange
- Use terse language to save tokens
- Do not add repetitive observations that have already been observed
- If the agent calls tools, observe what was called, why, and what was learned
- If the agent provides a detailed response, observe the contents so it could be repeated
- Make sure you start each observation with a priority emoji (🔴, 🟡, 🟢) or a completion marker (✅)
- Capture the user's words closely
- Treat ✅ as a memory signal that tells the assistant something is finished and should not be repeated unless new information changes it
- Observe WHAT the agent did and WHAT it means
- If the user provides detailed messages or code snippets, observe all important details"#;

const REFLECTOR_SYSTEM_PROMPT_PREFIX: &str = r#"You are the memory consciousness of an AI assistant. Your memory observation reflections will be the ONLY information the assistant has about past interactions with this user.

The following instructions were given to another part of your psyche (the observer) to create memories.
Use this to understand how your observational memories were created.

<observational-memory-instruction>"#;

const REFLECTOR_SYSTEM_PROMPT_SUFFIX: &str = r#"</observational-memory-instruction>

You are another part of the same psyche, the observation reflector.
Your reason for existing is to reflect on all the observations, re-organize and streamline them, and draw connections and conclusions between observations about what you've learned, seen, heard, and done.

IMPORTANT: your reflections are THE ENTIRETY of the assistants memory. Any information you do not add to your reflections will be immediately forgotten.

When consolidating observations:
- Preserve and include dates/times when present
- Retain the most relevant timestamps
- Combine related items where it makes sense
- Preserve ✅ completion markers and the concrete resolved outcomes they capture
- Condense older observations more aggressively, retain more detail for recent ones

CRITICAL: USER ASSERTIONS vs QUESTIONS
- "User stated: X" = authoritative assertion
- "User asked: X" = question/request
- When consolidating, USER ASSERTIONS TAKE PRECEDENCE

=== OUTPUT FORMAT ===

Your output MUST use XML tags to structure the response:

<observations>
Put all consolidated observations here using the date-grouped format with priority emojis (🔴, 🟡, 🟢).
Group related observations with indentation.
</observations>

<current-task>
State the current task(s) explicitly:
- Primary: What the agent is currently working on
- Secondary: Other pending tasks (mark as "waiting for user" if appropriate)
</current-task>

<suggested-response>
Hint for the agent's immediate next message.
</suggested-response>

User messages are extremely important. If the user asks a question or gives a new task, make it clear in <current-task> that this is the priority. If the assistant needs to respond to the user, indicate in <suggested-response> that it should pause for user reply before continuing other tasks."#;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ObservationBatchNode {
    observed_through_message_id: String,
    observations: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    current_task: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    suggested_response: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ReflectionNode {
    observed_through_message_id: String,
    observations: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    current_task: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    suggested_response: Option<String>,
}

#[derive(Clone, Debug)]
struct MemoryState {
    observations: String,
    current_task: Option<String>,
    suggested_response: Option<String>,
    observed_through_message_id: Option<String>,
    latest_memory_node_id: Option<String>,
}

#[derive(Clone, Debug)]
struct MessageNode {
    timestamp: String,
    message: Message,
}

#[derive(Clone, Debug, Default)]
struct ParsedMemoryOutput {
    observations: String,
    current_task: Option<String>,
    suggested_response: Option<String>,
}

#[derive(Default)]
pub struct ObservationalMemoryPluginFactory;

impl PluginFactory for ObservationalMemoryPluginFactory {
    fn id(&self) -> &'static str {
        OBSERVATIONAL_MEMORY_PLUGIN_ID
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        let ContextApproach::ObservationalMemory(config) = &ctx.context_approach else {
            return Ok(Arc::new(DisabledObservationalMemoryPlugin));
        };
        Ok(Arc::new(ObservationalMemoryPlugin {
            config: config.clone(),
        }))
    }
}

struct DisabledObservationalMemoryPlugin;

impl SessionPlugin for DisabledObservationalMemoryPlugin {
    fn id(&self) -> &'static str {
        OBSERVATIONAL_MEMORY_PLUGIN_ID
    }

    fn register(&self, _reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        Ok(())
    }
}

struct ObservationalMemoryPlugin {
    config: ObservationalMemoryConfig,
}

impl SessionPlugin for ObservationalMemoryPlugin {
    fn id(&self) -> &'static str {
        OBSERVATIONAL_MEMORY_PLUGIN_ID
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.history().prepare_turn(
            100,
            Arc::new(ObservationalMemoryTransform::new(self.config.clone())),
        );

        let config = self.config.clone();
        reg.session().on_event(Arc::new(move |event| {
            let config = config.clone();
            Box::pin(async move {
                if let PluginRuntimeEvent::TurnPersisted(ctx) = event {
                    let session_id = ctx.session_id.clone();
                    let host = Arc::clone(&ctx.host);
                    host.spawn_background_job(
                        &session_id,
                        OBSERVATIONAL_MEMORY_PLUGIN_ID,
                        Box::pin(async move {
                            if let Err(err) = run_async_maintenance(config, ctx).await {
                                tracing::warn!("observational-memory maintenance failed: {err}");
                            }
                            Ok(())
                        }),
                    )
                    .await?;
                }
                Ok(())
            })
        }));

        Ok(())
    }
}

struct ObservationalMemoryTransform;

impl ObservationalMemoryTransform {
    fn new(_config: ObservationalMemoryConfig) -> Self {
        Self
    }
}

#[async_trait]
impl TurnContextTransform for ObservationalMemoryTransform {
    fn id(&self) -> &'static str {
        "observational_memory.prepare_turn"
    }

    async fn transform(
        &self,
        ctx: &TurnTransformContext,
        input: PreparedContext,
    ) -> Result<PreparedContext, HistoryError> {
        let Some(memory_state) = build_memory_state(&ctx.state.session_graph) else {
            return Ok(input);
        };
        if memory_state.observations.trim().is_empty()
            && memory_state.current_task.is_none()
            && memory_state.suggested_response.is_none()
        {
            return Ok(input);
        }

        let prefix_len = input
            .messages
            .iter()
            .take_while(|message| matches!(message.role, MessageRole::System))
            .count();
        let tail_start = memory_state
            .observed_through_message_id
            .as_deref()
            .and_then(|message_id| {
                input
                    .messages
                    .iter()
                    .position(|message| message.id == message_id)
                    .map(|idx| idx + 1)
            })
            .unwrap_or(prefix_len);

        let mut messages = Vec::new();
        messages.extend_from_slice(&input.messages[..prefix_len]);
        messages.extend(build_memory_context_messages(&memory_state));
        messages.extend_from_slice(&input.messages[tail_start..]);

        Ok(PreparedContext { messages, ..input })
    }
}

fn debug_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let value = line.strip_prefix("VmRSS:")?.trim();
        let kb = value.split_whitespace().next()?.parse::<u64>().ok()?;
        Some(kb)
    })
}

async fn run_async_maintenance(
    config: ObservationalMemoryConfig,
    ctx: SessionStateChangedContext,
) -> Result<(), PluginError> {
    tracing::debug!(
        rss_kb = debug_rss_kb(),
        node_count = ctx.state.session_graph.nodes.len(),
        message_count = ctx.state.session_graph.project_messages().len(),
        "OM maintenance start"
    );
    let mut graph = ctx.state.session_graph.clone();
    if let Some(next) = maybe_run_observer_batches(
        &config,
        &ctx.session_id,
        &ctx.host,
        &ctx.state.policy,
        &graph,
    )
    .await?
    {
        graph = next;
        tracing::debug!(
            rss_kb = debug_rss_kb(),
            node_count = graph.nodes.len(),
            "OM maintenance after observer batches"
        );
    }
    let _ = maybe_run_reflector(
        &config,
        &ctx.session_id,
        &ctx.host,
        &ctx.state.policy,
        &graph,
    )
    .await?;
    tracing::debug!(
        rss_kb = debug_rss_kb(),
        node_count = graph.nodes.len(),
        "OM maintenance end"
    );
    Ok(())
}

async fn maybe_run_observer_batches(
    config: &ObservationalMemoryConfig,
    session_id: &str,
    host: &Arc<dyn SessionManager>,
    policy: &crate::SessionPolicy,
    graph: &SessionGraph,
) -> Result<Option<SessionGraph>, PluginError> {
    let memory_state = build_memory_state(graph);
    let unobserved = active_unobserved_message_nodes(graph, memory_state.as_ref());
    if unobserved.is_empty() {
        return Ok(None);
    }

    let total_tokens = approx_message_nodes_tokens(&unobserved);
    if total_tokens < config.observation_activation_tokens() {
        return Ok(None);
    }

    let observe_until =
        prefix_len_leaving_tail_budget(&unobserved, config.observation_buffer_tokens);
    if observe_until == 0 {
        return Ok(None);
    }

    let batches = split_message_batches(
        &unobserved[..observe_until],
        config.observation_max_tokens_per_batch.max(1),
    );
    if batches.is_empty() {
        return Ok(None);
    }
    tracing::debug!(
        rss_kb = debug_rss_kb(),
        unobserved_messages = unobserved.len(),
        observe_until,
        batch_count = batches.len(),
        total_tokens,
        "OM observer batches queued"
    );

    let mut next_graph = graph.clone();
    let mut next_memory_state = memory_state;
    let mut required_ancestor = next_graph.leaf_node_id.clone();

    for batch in batches {
        let output = run_observer_batch(
            config,
            session_id,
            host,
            policy.clone(),
            next_memory_state.as_ref(),
            &batch,
        )
        .await?;
        if output.observations.trim().is_empty()
            && output.current_task.is_none()
            && output.suggested_response.is_none()
        {
            continue;
        }

        let Some(last_message) = batch.last() else {
            continue;
        };
        let node = ObservationBatchNode {
            observed_through_message_id: last_message.message.id.clone(),
            observations: output.observations.trim().to_string(),
            current_task: output.current_task.clone(),
            suggested_response: output.suggested_response.clone(),
        };
        let body = serde_json::to_value(&node).map_err(|err| {
            PluginError::Snapshot(format!("failed to encode OM observation: {err}"))
        })?;
        match host
            .append_session_nodes(
                session_id,
                crate::AppendSessionNodesRequest {
                    nodes: vec![crate::SessionAppendNode::plugin(
                        OBSERVATION_PLUGIN_TYPE,
                        body,
                    )],
                    requires_ancestor_node_id: required_ancestor.clone(),
                },
            )
            .await?
        {
            crate::AppendSessionNodesResult::Appended { node_ids, .. } => {
                let Some(node_id) = node_ids.last().cloned() else {
                    continue;
                };
                next_graph.append_plugin(
                    OBSERVATION_PLUGIN_TYPE,
                    serde_json::to_value(&node).unwrap_or(json!(null)),
                );
                required_ancestor = Some(node_id.clone());
                next_memory_state = Some(merge_memory_state(
                    next_memory_state.as_ref(),
                    &node,
                    node_id,
                ));
            }
            crate::AppendSessionNodesResult::StaleBranch { .. } => break,
        }
    }

    Ok(Some(next_graph))
}

async fn maybe_run_reflector(
    config: &ObservationalMemoryConfig,
    session_id: &str,
    host: &Arc<dyn SessionManager>,
    policy: &crate::SessionPolicy,
    graph: &SessionGraph,
) -> Result<Option<SessionGraph>, PluginError> {
    let Some(memory_state) = build_memory_state(graph) else {
        return Ok(None);
    };
    if approx_token_count(&memory_state.observations) < config.reflection_activation_tokens() {
        return Ok(None);
    }

    let output =
        run_reflector(session_id, host, policy.clone(), &memory_state.observations).await?;
    if output.observations.trim().is_empty() {
        return Ok(None);
    }
    if output.observations.trim() == memory_state.observations.trim()
        && output.current_task == memory_state.current_task
        && output.suggested_response == memory_state.suggested_response
    {
        return Ok(None);
    }

    let Some(observed_through_message_id) = memory_state.observed_through_message_id.clone() else {
        return Ok(None);
    };
    let node = ReflectionNode {
        observed_through_message_id,
        observations: output.observations.trim().to_string(),
        current_task: output.current_task,
        suggested_response: output.suggested_response,
    };
    let body = serde_json::to_value(&node)
        .map_err(|err| PluginError::Snapshot(format!("failed to encode OM reflection: {err}")))?;
    match host
        .append_session_nodes(
            session_id,
            crate::AppendSessionNodesRequest {
                nodes: vec![crate::SessionAppendNode::plugin(
                    REFLECTION_PLUGIN_TYPE,
                    body,
                )],
                requires_ancestor_node_id: memory_state.latest_memory_node_id.clone(),
            },
        )
        .await?
    {
        crate::AppendSessionNodesResult::Appended { .. } => {
            let mut next_graph = graph.clone();
            next_graph.append_plugin(
                REFLECTION_PLUGIN_TYPE,
                serde_json::to_value(node).unwrap_or(json!(null)),
            );
            Ok(Some(next_graph))
        }
        crate::AppendSessionNodesResult::StaleBranch { .. } => Ok(None),
    }
}

async fn run_observer_batch(
    config: &ObservationalMemoryConfig,
    session_id: &str,
    host: &Arc<dyn SessionManager>,
    policy: crate::SessionPolicy,
    memory_state: Option<&MemoryState>,
    batch: &[MessageNode],
) -> Result<ParsedMemoryOutput, PluginError> {
    let existing_observations = memory_state
        .map(|state| {
            truncate_observation_tail(&state.observations, config.previous_observer_tokens)
        })
        .filter(|text| !text.trim().is_empty());
    let prior_current_task = memory_state.and_then(|state| state.current_task.clone());
    let prior_suggested_response = memory_state.and_then(|state| state.suggested_response.clone());
    let prompt = build_observer_prompt(
        existing_observations.as_deref(),
        batch,
        prior_current_task.as_deref(),
        prior_suggested_response.as_deref(),
    );
    run_worker_turn(
        session_id,
        host,
        policy,
        "observer",
        &observer_system_prompt(),
        &prompt,
    )
    .await
}

async fn run_reflector(
    session_id: &str,
    host: &Arc<dyn SessionManager>,
    policy: crate::SessionPolicy,
    observations: &str,
) -> Result<ParsedMemoryOutput, PluginError> {
    let prompt = build_reflector_prompt(observations);
    run_worker_turn(
        session_id,
        host,
        policy,
        "reflector",
        &reflector_system_prompt(),
        &prompt,
    )
    .await
}

async fn run_worker_turn(
    session_id: &str,
    host: &Arc<dyn SessionManager>,
    policy: crate::SessionPolicy,
    worker_kind: &str,
    system_prompt: &str,
    prompt: &str,
) -> Result<ParsedMemoryOutput, PluginError> {
    tracing::debug!(
        rss_kb = debug_rss_kb(),
        worker_kind,
        system_prompt_chars = system_prompt.len(),
        prompt_chars = prompt.len(),
        "OM worker direct completion start"
    );
    let completion = host
        .direct_completion(
            DirectRequest {
                model: policy.model,
                model_variant: policy.model_variant,
                messages: vec![
                    DirectMessage {
                        role: DirectRole::System,
                        parts: vec![DirectPart::Text(system_prompt.to_string())],
                    },
                    DirectMessage {
                        role: DirectRole::User,
                        parts: vec![DirectPart::Text(prompt.to_string())],
                    },
                ],
                attachments: Vec::new(),
                output: DirectOutputSpec::Text,
                stream_events: None,
                session_id: Some(format!("{session_id}-om-{worker_kind}")),
            },
            worker_kind,
        )
        .await;
    let completion = completion?;
    tracing::debug!(
        rss_kb = debug_rss_kb(),
        worker_kind,
        output_chars = completion.text.len(),
        "OM worker direct completion end"
    );
    Ok(parse_memory_output(&completion.text))
}

fn build_memory_context_messages(memory_state: &MemoryState) -> Vec<Message> {
    let mut messages = Vec::new();
    messages.push(plugin_message(
        "om-memory-system",
        MessageRole::System,
        format!("{OBSERVATION_CONTEXT_PROMPT}\n\n{OBSERVATION_CONTEXT_INSTRUCTIONS}"),
    ));

    let mut memory_block = String::from("<observations>\n");
    memory_block.push_str(memory_state.observations.trim());
    memory_block.push_str("\n</observations>");
    if let Some(current_task) = &memory_state.current_task {
        memory_block.push_str(&format!(
            "\n\n<current-task>\n{}\n</current-task>",
            current_task.trim()
        ));
    }
    if let Some(suggested_response) = &memory_state.suggested_response {
        memory_block.push_str(&format!(
            "\n\n<suggested-response>\n{}\n</suggested-response>",
            suggested_response.trim()
        ));
    }
    messages.push(plugin_message(
        "om-memory-block",
        MessageRole::System,
        memory_block,
    ));
    messages.push(plugin_message(
        "om-memory-reminder",
        MessageRole::User,
        format!("<system-reminder>{OBSERVATION_CONTINUATION_HINT}</system-reminder>"),
    ));
    messages
}

fn plugin_message(id: &str, role: MessageRole, content: String) -> Message {
    Message {
        id: id.to_string(),
        role,
        parts: vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Prose,
            content,
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            prune_state: crate::PruneState::Intact,
        }],
        user_input: None,
        origin: Some(MessageOrigin::Plugin {
            plugin_id: OBSERVATIONAL_MEMORY_PLUGIN_ID.to_string(),
            transient: true,
        }),
    }
}

fn build_memory_state(graph: &SessionGraph) -> Option<MemoryState> {
    let path = graph.active_path_nodes();
    let latest_reflection_idx = path.iter().rposition(|node| {
        matches!(
            node.plugin(),
            Some((kind, _)) if kind == REFLECTION_PLUGIN_TYPE
        )
    });

    let mut observations = Vec::new();
    let mut current_task = None;
    let mut suggested_response = None;
    let mut observed_through_message_id = None;
    let mut latest_memory_node_id = None;

    if let Some(idx) = latest_reflection_idx
        && let Some(node) = path.get(idx)
        && let Some((_, body)) = node.plugin()
        && let Ok(reflection) = serde_json::from_value::<ReflectionNode>(body.clone())
    {
        observations.push(reflection.observations);
        current_task = reflection.current_task;
        suggested_response = reflection.suggested_response;
        observed_through_message_id = Some(reflection.observed_through_message_id);
        latest_memory_node_id = Some(node.node_id.clone());
    }

    let start_idx = latest_reflection_idx.map_or(0, |idx| idx + 1);
    for node in path.into_iter().skip(start_idx) {
        let Some((kind, body)) = node.plugin() else {
            continue;
        };
        if kind != OBSERVATION_PLUGIN_TYPE {
            continue;
        }
        let Ok(batch) = serde_json::from_value::<ObservationBatchNode>(body.clone()) else {
            continue;
        };
        if !batch.observations.trim().is_empty() {
            observations.push(batch.observations);
        }
        if batch
            .current_task
            .as_ref()
            .is_some_and(|text| !text.trim().is_empty())
        {
            current_task = batch.current_task;
        }
        if batch
            .suggested_response
            .as_ref()
            .is_some_and(|text| !text.trim().is_empty())
        {
            suggested_response = batch.suggested_response;
        }
        observed_through_message_id = Some(batch.observed_through_message_id);
        latest_memory_node_id = Some(node.node_id.clone());
    }

    let observations = observations
        .into_iter()
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    if observations.trim().is_empty() && current_task.is_none() && suggested_response.is_none() {
        return None;
    }

    Some(MemoryState {
        observations,
        current_task,
        suggested_response,
        observed_through_message_id,
        latest_memory_node_id,
    })
}

fn active_unobserved_message_nodes(
    graph: &SessionGraph,
    memory_state: Option<&MemoryState>,
) -> Vec<MessageNode> {
    let observed_through =
        memory_state.and_then(|state| state.observed_through_message_id.as_deref());
    let mut seen_observed = observed_through.is_none();
    graph
        .active_path_nodes()
        .into_iter()
        .filter_map(|node| {
            let message = node.message()?.clone();
            if matches!(message.role, MessageRole::System) {
                return None;
            }
            if !seen_observed {
                if observed_through == Some(message.id.as_str()) {
                    seen_observed = true;
                }
                return None;
            }
            Some(MessageNode {
                timestamp: node.timestamp.clone(),
                message,
            })
        })
        .collect()
}

fn merge_memory_state(
    previous: Option<&MemoryState>,
    batch: &ObservationBatchNode,
    latest_memory_node_id: String,
) -> MemoryState {
    let mut observations = previous
        .map(|state| state.observations.clone())
        .unwrap_or_default();
    if !observations.trim().is_empty() && !batch.observations.trim().is_empty() {
        observations.push_str("\n\n");
    }
    observations.push_str(batch.observations.trim());
    MemoryState {
        observations,
        current_task: batch
            .current_task
            .clone()
            .or_else(|| previous.and_then(|state| state.current_task.clone())),
        suggested_response: batch
            .suggested_response
            .clone()
            .or_else(|| previous.and_then(|state| state.suggested_response.clone())),
        observed_through_message_id: Some(batch.observed_through_message_id.clone()),
        latest_memory_node_id: Some(latest_memory_node_id),
    }
}

fn prefix_len_leaving_tail_budget(messages: &[MessageNode], tail_budget_tokens: usize) -> usize {
    if messages.is_empty() {
        return 0;
    }
    if tail_budget_tokens == 0 {
        return messages.len();
    }
    let mut suffix_tokens = 0usize;
    for (idx, message) in messages.iter().enumerate().rev() {
        suffix_tokens = suffix_tokens.saturating_add(approx_message_tokens(message));
        if suffix_tokens > tail_budget_tokens {
            return idx + 1;
        }
    }
    0
}

fn split_message_batches(
    messages: &[MessageNode],
    max_tokens_per_batch: usize,
) -> Vec<Vec<MessageNode>> {
    let mut batches = Vec::new();
    let mut current = Vec::new();
    let mut current_tokens = 0usize;

    for message in messages {
        let tokens = approx_message_tokens(message).max(1);
        if !current.is_empty() && current_tokens + tokens > max_tokens_per_batch {
            batches.push(current);
            current = Vec::new();
            current_tokens = 0;
        }
        current.push(message.clone());
        current_tokens += tokens;
    }

    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

fn approx_message_nodes_tokens(messages: &[MessageNode]) -> usize {
    messages.iter().map(approx_message_tokens).sum()
}

fn approx_message_tokens(message: &MessageNode) -> usize {
    approx_token_count(&format_message_for_observer(message))
}

fn approx_token_count(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

fn truncate_observation_tail(observations: &str, budget_tokens: usize) -> String {
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
    format!("[Earlier observations omitted]\n{tail}")
}

fn build_observer_prompt(
    existing_observations: Option<&str>,
    messages: &[MessageNode],
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
        prompt.push_str("## Prior Thread Metadata\n\n");
        prompt.push_str(&prior_lines.join("\n"));
        prompt.push_str(
            "\n\nUse the prior current-task and suggested-response as continuity hints, then update them based on the new messages.\n\n---\n\n",
        );
    }
    prompt.push_str("## Your Task\n\nExtract new observations from the message history above. Do not repeat observations that are already in the previous observations. Add your new observations in the format specified in your instructions.");
    prompt
}

fn build_reflector_prompt(observations: &str) -> String {
    format!(
        "## OBSERVATIONS TO REFLECT ON\n\n{}\n\n---\n\nPlease analyze these observations and produce a refined, condensed version that will become the assistant's entire memory going forward.",
        observations.trim()
    )
}

fn format_message_for_observer(message: &MessageNode) -> String {
    let role = match message.message.role {
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::System => "system",
    };
    let mut lines = vec![format!("--- message boundary ({}) ---", message.timestamp)];
    lines.push(format!(
        "<message role=\"{role}\" id=\"{}\">",
        message.message.id
    ));
    for part in &message.message.parts {
        let kind = match part.kind {
            PartKind::Text => "text",
            PartKind::Image => "image",
            PartKind::Code => "code",
            PartKind::Output => "output",
            PartKind::Error => "error",
            PartKind::Prose => "prose",
            PartKind::ToolCall => "tool_call",
            PartKind::ToolResult => "tool_result",
        };
        let rendered = render_part_for_observer(part);
        if let Some(tool_name) = &part.tool_name {
            lines.push(format!(
                "<part kind=\"{kind}\" tool_name=\"{}\">",
                tool_name
            ));
        } else {
            lines.push(format!("<part kind=\"{kind}\">"));
        }
        lines.push(rendered);
        lines.push("</part>".to_string());
    }
    lines.push("</message>".to_string());
    lines.join("\n")
}

fn render_part_for_observer(part: &Part) -> String {
    if matches!(part.kind, PartKind::Image) {
        return if part.attachment.is_some() || part.content.trim().is_empty() {
            "[Image attached]".to_string()
        } else {
            part.content.clone()
        };
    }
    match &part.prune_state {
        crate::PruneState::Intact => part.content.clone(),
        crate::PruneState::Cleared => "[Old tool result content cleared]".to_string(),
        crate::PruneState::Deleted { breadcrumb, .. } => breadcrumb.clone(),
        crate::PruneState::Summarized { summary, .. } => summary.clone(),
    }
}

fn observer_system_prompt() -> String {
    format!(
        "You are the memory consciousness of an AI assistant. Your observations will be the ONLY information the assistant has about past interactions with this user.\n\nExtract observations that will help the assistant remember:\n\n{OBSERVER_EXTRACTION_INSTRUCTIONS}\n\n=== OUTPUT FORMAT ===\n\nYour output MUST use XML tags to structure the response. This allows the system to properly parse and manage memory over time.\n\n{OBSERVER_OUTPUT_FORMAT_BASE}\n\n=== GUIDELINES ===\n\n{OBSERVER_GUIDELINES}\n\n=== IMPORTANT: THREAD ATTRIBUTION ===\n\nDo NOT add thread identifiers, thread IDs, or <thread> tags to your observations.\nThread attribution is handled externally by the system.\nSimply output your observations without any thread-related markup.\n\nRemember: These observations are the assistant's ONLY memory. Make them count.\n\nUser messages are extremely important. If the user asks a question or gives a new task, make it clear in <current-task> that this is the priority. If the assistant needs to respond to the user, indicate in <suggested-response> that it should pause for user reply before continuing other tasks."
    )
}

fn reflector_system_prompt() -> String {
    format!(
        "{REFLECTOR_SYSTEM_PROMPT_PREFIX}\n{OBSERVER_EXTRACTION_INSTRUCTIONS}\n\n=== OUTPUT FORMAT ===\n\n{OBSERVER_OUTPUT_FORMAT_BASE}\n\n=== GUIDELINES ===\n\n{OBSERVER_GUIDELINES}\n{REFLECTOR_SYSTEM_PROMPT_SUFFIX}"
    )
}

fn parse_memory_output(output: &str) -> ParsedMemoryOutput {
    ParsedMemoryOutput {
        observations: extract_xml_block(output, "observations")
            .unwrap_or_else(|| output.trim().to_string()),
        current_task: extract_xml_block(output, "current-task")
            .filter(|text| !text.trim().is_empty()),
        suggested_response: extract_xml_block(output, "suggested-response")
            .filter(|text| !text.trim().is_empty()),
    }
}

fn extract_xml_block(content: &str, tag: &str) -> Option<String> {
    let pattern = format!(
        r"(?ims)^[ \t]*<{}>(.*?)^[ \t]*</{}>",
        regex::escape(tag),
        regex::escape(tag)
    );
    let regex = Regex::new(&pattern).ok()?;
    regex
        .captures(content)
        .and_then(|captures| captures.get(1))
        .map(|matched| matched.as_str().trim().to_string())
        .filter(|text| !text.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_message(id: &str, content: &str) -> MessageNode {
        MessageNode {
            timestamp: "2026-04-12T10:00:00Z".to_string(),
            message: Message {
                id: id.to_string(),
                role: MessageRole::User,
                parts: vec![Part {
                    id: format!("{id}.p0"),
                    kind: PartKind::Text,
                    content: content.to_string(),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: crate::PruneState::Intact,
                }],
                user_input: None,
                origin: None,
            },
        }
    }

    #[test]
    fn memory_state_prefers_latest_reflection_then_incremental_observations() {
        let mut graph = SessionGraph::default();
        graph.append_message(Message {
            id: "m1".to_string(),
            role: MessageRole::User,
            parts: vec![Part {
                id: "m1.p0".to_string(),
                kind: PartKind::Text,
                content: "hello".to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: crate::PruneState::Intact,
            }],
            user_input: None,
            origin: None,
        });
        graph.append_plugin(
            OBSERVATION_PLUGIN_TYPE,
            serde_json::to_value(ObservationBatchNode {
                observed_through_message_id: "m1".to_string(),
                observations: "Date: Apr 12, 2026\n* 🔴 User said hello".to_string(),
                current_task: Some("Greet the user".to_string()),
                suggested_response: None,
            })
            .expect("obs"),
        );
        graph.append_plugin(
            REFLECTION_PLUGIN_TYPE,
            serde_json::to_value(ReflectionNode {
                observed_through_message_id: "m1".to_string(),
                observations: "Date: Apr 12, 2026\n* 🔴 Reflected hello".to_string(),
                current_task: Some("Continue greeting".to_string()),
                suggested_response: Some("Respond warmly.".to_string()),
            })
            .expect("reflection"),
        );
        graph.append_message(Message {
            id: "m2".to_string(),
            role: MessageRole::User,
            parts: vec![Part {
                id: "m2.p0".to_string(),
                kind: PartKind::Text,
                content: "need help".to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: crate::PruneState::Intact,
            }],
            user_input: None,
            origin: None,
        });
        graph.append_plugin(
            OBSERVATION_PLUGIN_TYPE,
            serde_json::to_value(ObservationBatchNode {
                observed_through_message_id: "m2".to_string(),
                observations: "Date: Apr 12, 2026\n* 🔴 User asked for help".to_string(),
                current_task: Some("Help the user".to_string()),
                suggested_response: Some("Ask what they need.".to_string()),
            })
            .expect("obs2"),
        );

        let state = build_memory_state(&graph).expect("memory state");
        assert!(state.observations.contains("Reflected hello"));
        assert!(state.observations.contains("User asked for help"));
        assert_eq!(state.current_task.as_deref(), Some("Help the user"));
        assert_eq!(
            state.suggested_response.as_deref(),
            Some("Ask what they need.")
        );
        assert_eq!(state.observed_through_message_id.as_deref(), Some("m2"));
    }

    #[test]
    fn prefix_len_leaves_tail_budget() {
        let messages = vec![
            user_message("m1", &"a".repeat(4000)),
            user_message("m2", &"b".repeat(4000)),
            user_message("m3", &"c".repeat(4000)),
        ];
        let prefix = prefix_len_leaving_tail_budget(&messages, 1200);
        assert_eq!(prefix, 2);
    }

    #[test]
    fn parse_memory_output_extracts_xml_sections() {
        let parsed = parse_memory_output(
            "<observations>\nDate: Apr 12, 2026\n* 🔴 Test\n</observations>\n<current-task>\nWork\n</current-task>\n<suggested-response>\nContinue\n</suggested-response>",
        );
        assert!(parsed.observations.contains("Test"));
        assert_eq!(parsed.current_task.as_deref(), Some("Work"));
        assert_eq!(parsed.suggested_response.as_deref(), Some("Continue"));
    }
}
