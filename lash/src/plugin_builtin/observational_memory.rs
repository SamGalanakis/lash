use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::plugin::{
    HistoryError, PluginError, PluginFactory, PluginRegistrar, PluginRuntimeEvent,
    PluginSessionContext, SessionManager, SessionPlugin, TurnContextTransform,
    TurnTransformContext,
};
use crate::session_model::context::PreparedContext;
use crate::{
    AppendSessionNodesRequest, AppendSessionNodesResult, ContextApproach, DirectMessage,
    DirectOutputSpec, DirectPart, DirectRequest, DirectRole, Message, MessageOrigin, MessageRole,
    ObservationalMemoryConfig, Part, PartKind, SessionAppendNode, SessionGraph,
    SessionStateChangedContext,
};

const OBSERVATIONAL_MEMORY_PLUGIN_ID: &str = "observational_memory";
const ACTIVE_STATE_PLUGIN_TYPE: &str = "lash.context.observational_memory.state";
const BUFFERED_OBSERVATION_PLUGIN_TYPE: &str =
    "lash.context.observational_memory.buffered_observation";
const BUFFERED_REFLECTION_PLUGIN_TYPE: &str =
    "lash.context.observational_memory.buffered_reflection";

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
struct ActiveMemoryNode {
    observed_through_message_id: String,
    observations: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    current_task: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    suggested_response: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BufferedObservationNode {
    observed_through_message_id: String,
    observations: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    current_task: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    suggested_response: Option<String>,
    #[serde(default)]
    observation_tokens: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BufferedReflectionNode {
    source_state_node_id: String,
    observed_through_message_id: String,
    observations: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    current_task: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    suggested_response: Option<String>,
    #[serde(default)]
    observation_tokens: usize,
}

#[derive(Clone, Debug)]
struct ActiveMemoryState {
    state_node_id: String,
    observed_through_message_id: Option<String>,
    observations: String,
    current_task: Option<String>,
    suggested_response: Option<String>,
}

#[derive(Clone, Debug)]
struct BufferedObservationState {
    observed_through_message_id: String,
    observations: String,
    current_task: Option<String>,
    suggested_response: Option<String>,
}

#[derive(Clone, Debug)]
struct BufferedReflectionState {
    source_state_node_id: String,
    observed_through_message_id: String,
    observations: String,
    current_task: Option<String>,
    suggested_response: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct OmGraphState {
    active: Option<ActiveMemoryState>,
    buffered_observations: Vec<BufferedObservationState>,
    buffered_reflection: Option<BufferedReflectionState>,
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct MessageNode {
    timestamp: String,
    message: Message,
}

#[derive(Clone, Copy, Debug)]
struct MessageNodeRef<'a> {
    timestamp: &'a str,
    message: &'a Message,
}

trait ObservedMessageNode {
    fn timestamp(&self) -> &str;
    fn message(&self) -> &Message;
}

#[cfg(test)]
impl ObservedMessageNode for MessageNode {
    fn timestamp(&self) -> &str {
        &self.timestamp
    }

    fn message(&self) -> &Message {
        &self.message
    }
}

impl<'a> ObservedMessageNode for MessageNodeRef<'a> {
    fn timestamp(&self) -> &str {
        self.timestamp
    }

    fn message(&self) -> &Message {
        self.message
    }
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

    fn supported_context_approaches(&self) -> &'static [crate::ContextApproachKind] {
        &[crate::ContextApproachKind::ObservationalMemory]
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
                    let graph = ctx.state.session_graph().clone();
                    if !should_run_async_maintenance(&config, &graph) {
                        return Ok(());
                    }
                    let session_id = ctx.session_id.clone();
                    let host = Arc::clone(&ctx.host);
                    host.spawn_hidden_task(
                        &session_id,
                        OBSERVATIONAL_MEMORY_PLUGIN_ID,
                        Box::pin(async move {
                            if let Err(err) = run_async_maintenance(config, graph, ctx).await {
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

struct ObservationalMemoryTransform {
    config: ObservationalMemoryConfig,
}

impl ObservationalMemoryTransform {
    fn new(config: ObservationalMemoryConfig) -> Self {
        Self { config }
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
        let mut graph = ctx.state.session_graph().clone();
        let om_state = build_graph_state(&graph);
        let pending_message_tokens = approx_message_nodes_tokens(&active_unobserved_message_nodes(
            &graph,
            om_state
                .active
                .as_ref()
                .and_then(|state| state.observed_through_message_id.as_deref()),
        ));
        let active_observation_tokens = om_state
            .active
            .as_ref()
            .map(|state| approx_token_count(&state.observations))
            .unwrap_or(0);

        if pending_message_tokens >= self.config.observation_message_tokens
            || active_observation_tokens >= self.config.reflection_observation_tokens
        {
            ctx.host.await_hidden_tasks(&ctx.session_id).await?;
            graph = ctx.host.snapshot_current().await?.session_graph;
        }

        graph = maybe_advance_memory_state(
            &self.config,
            &ctx.session_id,
            &ctx.host,
            ctx.state.policy(),
            graph,
        )
        .await?;

        let Some(active) = build_graph_state(&graph).active else {
            return Ok(input);
        };
        if active.observations.trim().is_empty()
            && active.current_task.is_none()
            && active.suggested_response.is_none()
        {
            return Ok(input);
        }

        let input_messages = input.messages.as_slice();
        let prefix_len = input_messages
            .iter()
            .take_while(|message| matches!(message.role, MessageRole::System))
            .count();
        let tail_start = active
            .observed_through_message_id
            .as_deref()
            .and_then(|message_id| {
                input_messages
                    .iter()
                    .position(|message| message.id == message_id)
                    .map(|idx| idx + 1)
            })
            .unwrap_or(prefix_len);

        let mut messages = Vec::new();
        messages.extend_from_slice(&input_messages[..prefix_len]);
        messages.extend(build_memory_context_messages(&active));
        messages.extend_from_slice(&input_messages[tail_start..]);

        Ok(PreparedContext {
            messages: crate::MessageSequence::from_owned(messages),
            ..input
        })
    }
}

async fn run_async_maintenance(
    config: ObservationalMemoryConfig,
    graph: SessionGraph,
    ctx: SessionStateChangedContext,
) -> Result<(), PluginError> {
    maybe_buffer_observations(
        &config,
        &ctx.session_id,
        &ctx.host,
        ctx.state.policy(),
        &graph,
    )
    .await?;
    maybe_buffer_reflection(
        &config,
        &ctx.session_id,
        &ctx.host,
        ctx.state.policy(),
        &graph,
    )
    .await?;
    Ok(())
}

fn should_run_async_maintenance(config: &ObservationalMemoryConfig, graph: &SessionGraph) -> bool {
    let om_state = build_graph_state(graph);
    let start_after = om_state
        .buffered_observations
        .last()
        .map(|chunk| chunk.observed_through_message_id.as_str())
        .or_else(|| {
            om_state
                .active
                .as_ref()
                .and_then(|state| state.observed_through_message_id.as_deref())
        });
    let observation_interval = config.observation_buffer_interval_tokens();
    if observation_interval > 0
        && approx_message_nodes_tokens(&active_unobserved_message_nodes(graph, start_after))
            >= observation_interval
    {
        return true;
    }

    om_state.buffered_reflection.is_none()
        && om_state
            .active
            .as_ref()
            .map(|active| approx_token_count(&active.observations))
            .unwrap_or(0)
            >= config.reflection_buffer_activation_tokens()
}

async fn maybe_advance_memory_state(
    config: &ObservationalMemoryConfig,
    session_id: &str,
    host: &Arc<dyn SessionManager>,
    policy: &crate::SessionPolicy,
    mut graph: SessionGraph,
) -> Result<SessionGraph, PluginError> {
    for _ in 0..6 {
        let om_state = build_graph_state(&graph);
        let pending_messages = active_unobserved_message_nodes(
            &graph,
            om_state
                .active
                .as_ref()
                .and_then(|state| state.observed_through_message_id.as_deref()),
        );
        let pending_tokens = approx_message_nodes_tokens(&pending_messages);
        if pending_tokens >= config.observation_message_tokens {
            if let Some(next) =
                activate_buffered_observations(config, session_id, host, &graph).await?
            {
                graph = next;
                continue;
            }
            if pending_tokens >= config.observation_block_after_tokens
                && let Some(next) =
                    sync_observe_pending_messages(config, session_id, host, policy, &graph).await?
            {
                graph = next;
                continue;
            }
        }

        let om_state = build_graph_state(&graph);
        let active_tokens = om_state
            .active
            .as_ref()
            .map(|state| approx_token_count(&state.observations))
            .unwrap_or(0);
        if active_tokens >= config.reflection_observation_tokens {
            if let Some(next) = activate_buffered_reflection(session_id, host, &graph).await? {
                graph = next;
                continue;
            }
            if active_tokens >= config.reflection_block_after_tokens
                && let Some(next) =
                    sync_reflect_active_memory(session_id, host, policy, &graph).await?
            {
                graph = next;
                continue;
            }
        }

        break;
    }

    Ok(graph)
}

async fn maybe_buffer_observations(
    config: &ObservationalMemoryConfig,
    session_id: &str,
    host: &Arc<dyn SessionManager>,
    policy: &crate::SessionPolicy,
    graph: &SessionGraph,
) -> Result<(), PluginError> {
    let om_state = build_graph_state(graph);
    let start_after = om_state
        .buffered_observations
        .last()
        .map(|chunk| chunk.observed_through_message_id.as_str())
        .or_else(|| {
            om_state
                .active
                .as_ref()
                .and_then(|state| state.observed_through_message_id.as_deref())
        });
    let unbuffered = active_unobserved_message_nodes(graph, start_after);
    let interval_tokens = config.observation_buffer_interval_tokens();
    if interval_tokens == 0 {
        return Ok(());
    }
    let total_tokens = approx_message_nodes_tokens(&unbuffered);
    if total_tokens < interval_tokens {
        return Ok(());
    }

    let target_tokens = total_tokens - (total_tokens % interval_tokens);
    let Some(prefix_len) = prefix_len_covering_tokens(&unbuffered, target_tokens) else {
        return Ok(());
    };
    if prefix_len == 0 {
        return Ok(());
    }
    let batch_target = config
        .observation_max_tokens_per_batch
        .min(interval_tokens)
        .max(1);
    let batches = split_message_batches(&unbuffered[..prefix_len], batch_target);
    if batches.is_empty() {
        return Ok(());
    }

    let mut preview = om_state.active.clone();
    let mut nodes = Vec::new();
    for batch in batches {
        let output = run_observer_batch(
            config,
            session_id,
            host,
            policy.clone(),
            preview.as_ref(),
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
        let node = BufferedObservationNode {
            observed_through_message_id: last_message.message.id.clone(),
            observations: output.observations.trim().to_string(),
            current_task: output.current_task.clone(),
            suggested_response: output.suggested_response.clone(),
            observation_tokens: approx_token_count(output.observations.trim()),
        };
        preview = Some(merge_active_state_with_observation(preview.as_ref(), &node));
        nodes.push((
            BUFFERED_OBSERVATION_PLUGIN_TYPE.to_string(),
            serde_json::to_value(node).map_err(|err| {
                PluginError::Snapshot(format!("failed to encode OM buffered observation: {err}"))
            })?,
        ));
    }

    if nodes.is_empty() {
        return Ok(());
    }

    let _ = append_plugin_nodes(session_id, host, graph, nodes).await?;
    Ok(())
}

async fn maybe_buffer_reflection(
    config: &ObservationalMemoryConfig,
    session_id: &str,
    host: &Arc<dyn SessionManager>,
    policy: &crate::SessionPolicy,
    graph: &SessionGraph,
) -> Result<(), PluginError> {
    let om_state = build_graph_state(graph);
    let Some(active) = om_state.active else {
        return Ok(());
    };
    if om_state.buffered_reflection.is_some() {
        return Ok(());
    }

    let observation_tokens = approx_token_count(&active.observations);
    if observation_tokens < config.reflection_buffer_activation_tokens() {
        return Ok(());
    }

    let output = run_reflector(session_id, host, policy.clone(), &active.observations).await?;
    if output.observations.trim().is_empty() {
        return Ok(());
    }
    if output.observations.trim() == active.observations.trim()
        && output.current_task == active.current_task
        && output.suggested_response == active.suggested_response
    {
        return Ok(());
    }

    let node = BufferedReflectionNode {
        source_state_node_id: active.state_node_id,
        observed_through_message_id: active.observed_through_message_id.unwrap_or_default(),
        observations: output.observations.trim().to_string(),
        current_task: output.current_task,
        suggested_response: output.suggested_response,
        observation_tokens,
    };
    let _ = append_plugin_nodes(
        session_id,
        host,
        graph,
        vec![(
            BUFFERED_REFLECTION_PLUGIN_TYPE.to_string(),
            serde_json::to_value(node).map_err(|err| {
                PluginError::Snapshot(format!("failed to encode OM buffered reflection: {err}"))
            })?,
        )],
    )
    .await?;
    Ok(())
}

async fn activate_buffered_observations(
    config: &ObservationalMemoryConfig,
    session_id: &str,
    host: &Arc<dyn SessionManager>,
    graph: &SessionGraph,
) -> Result<Option<SessionGraph>, PluginError> {
    let om_state = build_graph_state(graph);
    if om_state.buffered_observations.is_empty() {
        return Ok(None);
    }
    let pending_messages = active_unobserved_message_nodes(
        graph,
        om_state
            .active
            .as_ref()
            .and_then(|state| state.observed_through_message_id.as_deref()),
    );
    if pending_messages.is_empty() {
        return Ok(None);
    }

    let retained_after = retained_message_tokens_by_message_id(&pending_messages);
    let mut activated = Vec::new();
    let mut merged = om_state.active.clone();
    for chunk in &om_state.buffered_observations {
        activated.push(chunk.clone());
        merged = Some(merge_active_state_with_observation(
            merged.as_ref(),
            &BufferedObservationNode {
                observed_through_message_id: chunk.observed_through_message_id.clone(),
                observations: chunk.observations.clone(),
                current_task: chunk.current_task.clone(),
                suggested_response: chunk.suggested_response.clone(),
                observation_tokens: approx_token_count(&chunk.observations),
            },
        ));
        let remaining = retained_after
            .get(chunk.observed_through_message_id.as_str())
            .copied()
            .unwrap_or(0);
        if remaining <= config.observation_retention_tokens() {
            break;
        }
    }

    if activated.is_empty() {
        return Ok(None);
    }

    let Some(next_active) = merged else {
        return Ok(None);
    };
    let node = ActiveMemoryNode {
        observed_through_message_id: next_active.observed_through_message_id.unwrap_or_default(),
        observations: next_active.observations,
        current_task: next_active.current_task,
        suggested_response: next_active.suggested_response,
    };
    match append_plugin_nodes(
        session_id,
        host,
        graph,
        vec![(
            ACTIVE_STATE_PLUGIN_TYPE.to_string(),
            serde_json::to_value(node).map_err(|err| {
                PluginError::Snapshot(format!("failed to encode OM state activation: {err}"))
            })?,
        )],
    )
    .await?
    {
        Some(next) => Ok(Some(next)),
        None => Ok(None),
    }
}

async fn activate_buffered_reflection(
    session_id: &str,
    host: &Arc<dyn SessionManager>,
    graph: &SessionGraph,
) -> Result<Option<SessionGraph>, PluginError> {
    let om_state = build_graph_state(graph);
    let Some(active) = om_state.active else {
        return Ok(None);
    };
    let Some(buffered) = om_state.buffered_reflection else {
        return Ok(None);
    };
    if buffered.source_state_node_id != active.state_node_id {
        return Ok(None);
    }
    let node = ActiveMemoryNode {
        observed_through_message_id: buffered.observed_through_message_id,
        observations: buffered.observations,
        current_task: buffered.current_task,
        suggested_response: buffered.suggested_response,
    };
    append_plugin_nodes(
        session_id,
        host,
        graph,
        vec![(
            ACTIVE_STATE_PLUGIN_TYPE.to_string(),
            serde_json::to_value(node).map_err(|err| {
                PluginError::Snapshot(format!("failed to encode OM reflection activation: {err}"))
            })?,
        )],
    )
    .await
}

async fn sync_observe_pending_messages(
    config: &ObservationalMemoryConfig,
    session_id: &str,
    host: &Arc<dyn SessionManager>,
    policy: &crate::SessionPolicy,
    graph: &SessionGraph,
) -> Result<Option<SessionGraph>, PluginError> {
    let om_state = build_graph_state(graph);
    let unobserved = active_unobserved_message_nodes(
        graph,
        om_state
            .active
            .as_ref()
            .and_then(|state| state.observed_through_message_id.as_deref()),
    );
    if unobserved.is_empty() {
        return Ok(None);
    }
    let observe_until =
        prefix_len_leaving_tail_budget(&unobserved, config.observation_retention_tokens());
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

    let mut merged = om_state.active.clone();
    for batch in batches {
        let output = run_observer_batch(
            config,
            session_id,
            host,
            policy.clone(),
            merged.as_ref(),
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
        merged = Some(merge_active_state_with_observer_output(
            merged.as_ref(),
            last_message.message.id.clone(),
            output,
        ));
    }

    let Some(active) = merged else {
        return Ok(None);
    };
    let node = ActiveMemoryNode {
        observed_through_message_id: active.observed_through_message_id.unwrap_or_default(),
        observations: active.observations,
        current_task: active.current_task,
        suggested_response: active.suggested_response,
    };
    append_plugin_nodes(
        session_id,
        host,
        graph,
        vec![(
            ACTIVE_STATE_PLUGIN_TYPE.to_string(),
            serde_json::to_value(node).map_err(|err| {
                PluginError::Snapshot(format!("failed to encode OM sync observation: {err}"))
            })?,
        )],
    )
    .await
}

async fn sync_reflect_active_memory(
    session_id: &str,
    host: &Arc<dyn SessionManager>,
    policy: &crate::SessionPolicy,
    graph: &SessionGraph,
) -> Result<Option<SessionGraph>, PluginError> {
    let om_state = build_graph_state(graph);
    let Some(active) = om_state.active else {
        return Ok(None);
    };
    let output = run_reflector(session_id, host, policy.clone(), &active.observations).await?;
    if output.observations.trim().is_empty() {
        return Ok(None);
    }
    if output.observations.trim() == active.observations.trim()
        && output.current_task == active.current_task
        && output.suggested_response == active.suggested_response
    {
        return Ok(None);
    }
    let node = ActiveMemoryNode {
        observed_through_message_id: active.observed_through_message_id.unwrap_or_default(),
        observations: output.observations.trim().to_string(),
        current_task: output.current_task,
        suggested_response: output.suggested_response,
    };
    append_plugin_nodes(
        session_id,
        host,
        graph,
        vec![(
            ACTIVE_STATE_PLUGIN_TYPE.to_string(),
            serde_json::to_value(node).map_err(|err| {
                PluginError::Snapshot(format!("failed to encode OM sync reflection: {err}"))
            })?,
        )],
    )
    .await
}

async fn append_plugin_nodes(
    session_id: &str,
    host: &Arc<dyn SessionManager>,
    graph: &SessionGraph,
    nodes: Vec<(String, serde_json::Value)>,
) -> Result<Option<SessionGraph>, PluginError> {
    if nodes.is_empty() {
        return Ok(Some(graph.clone()));
    }
    let request_nodes = nodes
        .iter()
        .cloned()
        .map(|(plugin_type, body)| SessionAppendNode::plugin(plugin_type, body))
        .collect::<Vec<_>>();
    match host
        .append_session_nodes(
            session_id,
            AppendSessionNodesRequest {
                nodes: request_nodes,
                requires_ancestor_node_id: graph.leaf_node_id.clone(),
            },
        )
        .await?
    {
        AppendSessionNodesResult::Appended { .. } => {
            let mut next = graph.clone();
            for (plugin_type, body) in nodes {
                next.append_plugin(plugin_type, body);
            }
            Ok(Some(next))
        }
        AppendSessionNodesResult::StaleBranch { .. } => Ok(None),
    }
}

fn merge_active_state_with_observation(
    previous: Option<&ActiveMemoryState>,
    batch: &BufferedObservationNode,
) -> ActiveMemoryState {
    let mut observations = previous
        .map(|state| state.observations.clone())
        .unwrap_or_default();
    if !observations.trim().is_empty() && !batch.observations.trim().is_empty() {
        observations.push_str("\n\n");
    }
    observations.push_str(batch.observations.trim());
    ActiveMemoryState {
        state_node_id: previous
            .map(|state| state.state_node_id.clone())
            .unwrap_or_default(),
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
    }
}

fn merge_active_state_with_observer_output(
    previous: Option<&ActiveMemoryState>,
    observed_through_message_id: String,
    output: ParsedMemoryOutput,
) -> ActiveMemoryState {
    let mut observations = previous
        .map(|state| state.observations.clone())
        .unwrap_or_default();
    if !observations.trim().is_empty() && !output.observations.trim().is_empty() {
        observations.push_str("\n\n");
    }
    observations.push_str(output.observations.trim());
    ActiveMemoryState {
        state_node_id: previous
            .map(|state| state.state_node_id.clone())
            .unwrap_or_default(),
        observations,
        current_task: output
            .current_task
            .or_else(|| previous.and_then(|state| state.current_task.clone())),
        suggested_response: output
            .suggested_response
            .or_else(|| previous.and_then(|state| state.suggested_response.clone())),
        observed_through_message_id: Some(observed_through_message_id),
    }
}

fn build_memory_context_messages(active: &ActiveMemoryState) -> Vec<Message> {
    let mut messages = Vec::new();
    messages.push(plugin_message(
        "om-memory-system",
        MessageRole::System,
        format!("{OBSERVATION_CONTEXT_PROMPT}\n\n{OBSERVATION_CONTEXT_INSTRUCTIONS}"),
    ));

    let mut memory_block = String::from("<observations>\n");
    memory_block.push_str(active.observations.trim());
    memory_block.push_str("\n</observations>");
    if let Some(current_task) = &active.current_task {
        memory_block.push_str(&format!(
            "\n\n<current-task>\n{}\n</current-task>",
            current_task.trim()
        ));
    }
    if let Some(suggested_response) = &active.suggested_response {
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

fn build_graph_state(graph: &SessionGraph) -> OmGraphState {
    let mut state = OmGraphState::default();
    for node in graph.active_path_nodes() {
        let Some((kind, _)) = node.plugin() else {
            continue;
        };
        match kind {
            ACTIVE_STATE_PLUGIN_TYPE => {
                let Some(active) = node.plugin_body::<ActiveMemoryNode>() else {
                    continue;
                };
                state.active = Some(ActiveMemoryState {
                    state_node_id: node.node_id.clone(),
                    observed_through_message_id: Some(active.observed_through_message_id),
                    observations: active.observations,
                    current_task: active.current_task,
                    suggested_response: active.suggested_response,
                });
                state.buffered_observations.clear();
                state.buffered_reflection = None;
            }
            BUFFERED_OBSERVATION_PLUGIN_TYPE => {
                let Some(buffered) = node.plugin_body::<BufferedObservationNode>() else {
                    continue;
                };
                if state.buffered_observations.iter().any(|chunk| {
                    chunk.observed_through_message_id == buffered.observed_through_message_id
                }) {
                    continue;
                }
                state.buffered_observations.push(BufferedObservationState {
                    observed_through_message_id: buffered.observed_through_message_id,
                    observations: buffered.observations,
                    current_task: buffered.current_task,
                    suggested_response: buffered.suggested_response,
                });
            }
            BUFFERED_REFLECTION_PLUGIN_TYPE => {
                let Some(buffered) = node.plugin_body::<BufferedReflectionNode>() else {
                    continue;
                };
                let Some(active) = state.active.as_ref() else {
                    continue;
                };
                if buffered.source_state_node_id != active.state_node_id {
                    continue;
                }
                state.buffered_reflection = Some(BufferedReflectionState {
                    source_state_node_id: buffered.source_state_node_id,
                    observed_through_message_id: buffered.observed_through_message_id,
                    observations: buffered.observations,
                    current_task: buffered.current_task,
                    suggested_response: buffered.suggested_response,
                });
            }
            _ => {}
        }
    }
    state
}

fn active_unobserved_message_nodes<'a>(
    graph: &'a SessionGraph,
    observed_through_message_id: Option<&str>,
) -> Vec<MessageNodeRef<'a>> {
    let mut seen_observed = observed_through_message_id.is_none();
    graph
        .active_path_nodes()
        .into_iter()
        .filter_map(|node| {
            let message = node.message()?;
            if matches!(message.role, MessageRole::System) {
                return None;
            }
            if !seen_observed {
                if observed_through_message_id == Some(message.id.as_str()) {
                    seen_observed = true;
                }
                return None;
            }
            Some(MessageNodeRef {
                timestamp: node.timestamp.as_str(),
                message,
            })
        })
        .collect()
}

fn retained_message_tokens_by_message_id<N: ObservedMessageNode>(
    messages: &[N],
) -> std::collections::HashMap<&str, usize> {
    let mut retained = std::collections::HashMap::new();
    let mut suffix_tokens = 0usize;
    for message in messages.iter().rev() {
        retained.insert(message.message().id.as_str(), suffix_tokens);
        suffix_tokens = suffix_tokens.saturating_add(approx_message_tokens(message));
    }
    retained
}

fn prefix_len_leaving_tail_budget<N: ObservedMessageNode>(
    messages: &[N],
    tail_budget_tokens: usize,
) -> usize {
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

fn prefix_len_covering_tokens<N: ObservedMessageNode>(
    messages: &[N],
    target_tokens: usize,
) -> Option<usize> {
    if target_tokens == 0 {
        return Some(0);
    }
    let mut total = 0usize;
    for (idx, message) in messages.iter().enumerate() {
        total = total.saturating_add(approx_message_tokens(message));
        if total >= target_tokens {
            return Some(idx + 1);
        }
    }
    None
}

fn split_message_batches<N: ObservedMessageNode + Clone>(
    messages: &[N],
    max_tokens_per_batch: usize,
) -> Vec<Vec<N>> {
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

fn approx_message_nodes_tokens<N: ObservedMessageNode>(messages: &[N]) -> usize {
    messages.iter().map(approx_message_tokens).sum()
}

fn approx_message_tokens<N: ObservedMessageNode>(message: &N) -> usize {
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

async fn run_observer_batch(
    config: &ObservationalMemoryConfig,
    session_id: &str,
    host: &Arc<dyn SessionManager>,
    policy: crate::SessionPolicy,
    active: Option<&ActiveMemoryState>,
    batch: &[impl ObservedMessageNode],
) -> Result<ParsedMemoryOutput, PluginError> {
    let existing_observations = active
        .map(|state| {
            truncate_observation_tail(&state.observations, config.previous_observer_tokens)
        })
        .filter(|text| !text.trim().is_empty());
    let prior_current_task = active.and_then(|state| state.current_task.clone());
    let prior_suggested_response = active.and_then(|state| state.suggested_response.clone());
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
        .await?;
    Ok(parse_memory_output(&completion.text))
}

fn build_observer_prompt(
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

fn build_reflector_prompt(observations: &str) -> String {
    format!(
        "## Observations to Reflect\n\n{}\n\n## Task\nRestructure and compress these observations while preserving the most important user facts, active tasks, and continuity cues.",
        observations.trim()
    )
}

fn observer_system_prompt() -> String {
    format!(
        "{}\n\n{}\n\n{}\n\n{}",
        OBSERVER_EXTRACTION_INSTRUCTIONS,
        OBSERVER_OUTPUT_FORMAT_BASE,
        OBSERVER_GUIDELINES,
        "Output only the XML blocks."
    )
}

fn reflector_system_prompt() -> String {
    format!(
        "{}\n{}\n\n{}",
        REFLECTOR_SYSTEM_PROMPT_PREFIX,
        observer_system_prompt(),
        REFLECTOR_SYSTEM_PROMPT_SUFFIX
    )
}

fn parse_memory_output(text: &str) -> ParsedMemoryOutput {
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

fn capture_xml_block(text: &str, tag: &str) -> Option<String> {
    let pattern = format!(r"(?s)<{tag}>\s*(.*?)\s*</{tag}>");
    let re = Regex::new(&pattern).ok()?;
    let captures = re.captures(text)?;
    Some(captures.get(1)?.as_str().to_string())
}

fn format_message_for_observer(node: &impl ObservedMessageNode) -> String {
    let timestamp = format_observation_timestamp(node.timestamp());
    let role = match node.message().role {
        MessageRole::User => "USER",
        MessageRole::Assistant => "ASSISTANT",
        MessageRole::System => "SYSTEM",
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
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("[{timestamp}] {role}\n{}", content.trim())
}

fn format_observation_timestamp(timestamp: &str) -> String {
    chrono::DateTime::parse_from_rfc3339(timestamp)
        .map(|dt| dt.format("%b %-d, %Y %H:%M").to_string())
        .unwrap_or_else(|_| timestamp.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_message(id: &str, content: &str) -> MessageNode {
        MessageNode {
            timestamp: "2026-04-14T10:00:00Z".to_string(),
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
    fn build_graph_state_resets_buffers_after_active_state() {
        let mut graph = SessionGraph::default();
        graph.append_message(user_message("m1", "hello").message);
        graph.append_plugin(
            BUFFERED_OBSERVATION_PLUGIN_TYPE,
            serde_json::json!({
                "observed_through_message_id": "m1",
                "observations": "old buffered",
                "observation_tokens": 10
            }),
        );
        graph.append_plugin(
            ACTIVE_STATE_PLUGIN_TYPE,
            serde_json::json!({
                "observed_through_message_id": "m1",
                "observations": "active memory"
            }),
        );
        graph.append_message(user_message("m2", "need help").message);
        graph.append_plugin(
            BUFFERED_OBSERVATION_PLUGIN_TYPE,
            serde_json::json!({
                "observed_through_message_id": "m2",
                "observations": "new buffered",
                "observation_tokens": 20
            }),
        );

        let state = build_graph_state(&graph);
        assert_eq!(
            state.active.as_ref().map(|item| item.observations.as_str()),
            Some("active memory")
        );
        assert_eq!(state.buffered_observations.len(), 1);
        assert_eq!(
            state.buffered_observations[0].observations,
            "new buffered".to_string()
        );
    }

    #[test]
    fn retained_message_tokens_tracks_suffix_after_message() {
        let messages = vec![
            user_message("m1", &"a".repeat(4000)),
            user_message("m2", &"b".repeat(4000)),
            user_message("m3", &"c".repeat(4000)),
        ];
        let retained = retained_message_tokens_by_message_id(&messages);
        assert_eq!(retained.get("m3").copied(), Some(0));
        assert!(retained.get("m2").copied().unwrap_or_default() > 0);
        assert!(retained.get("m1").copied().unwrap_or_default() > retained["m2"]);
    }

    #[test]
    fn prefix_len_covering_tokens_handles_partial_prefix() {
        let messages = vec![
            user_message("m1", &"a".repeat(4000)),
            user_message("m2", &"b".repeat(4000)),
            user_message("m3", &"c".repeat(4000)),
        ];
        let prefix = prefix_len_covering_tokens(&messages, 2000).expect("prefix");
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
