pub const OBSERVATIONAL_MEMORY_PLUGIN_ID: &str = "observational_memory";
pub const ACTIVE_STATE_PLUGIN_TYPE: &str = "lash.context.observational_memory.state";
pub const BUFFERED_OBSERVATION_PLUGIN_TYPE: &str =
    "lash.context.observational_memory.buffered_observation";
pub const BUFFERED_REFLECTION_PLUGIN_TYPE: &str =
    "lash.context.observational_memory.buffered_reflection";

pub(crate) const OBSERVATION_CONTEXT_PROMPT: &str =
    "The following observations block contains your memory of past conversations with this user.";
pub(crate) const OBSERVATION_CONTEXT_INSTRUCTIONS: &str = "IMPORTANT: When responding, reference specific details from these observations. Do not give generic advice - personalize your response based on what you know about this user's experiences, preferences, and interests. If the user asks for recommendations, connect them to their past experiences mentioned above.\n\nKNOWLEDGE UPDATES: When asked about current state (e.g., \"where do I currently...\", \"what is my current...\"), always prefer the MOST RECENT information. Observations include dates - if you see conflicting information, the newer observation supersedes the older one. Look for phrases like \"will start\", \"is switching\", \"changed to\", \"moved to\" as indicators that previous information has been updated.\n\nPLANNED ACTIONS: If the user stated they planned to do something (e.g., \"I'm going to...\", \"I'm looking forward to...\", \"I will...\") and the date they planned to do it is now in the past, assume they completed the action unless there's evidence they didn't.\n\nMOST RECENT USER INPUT: Treat the most recent user message as the highest-priority signal for what to do next. Earlier messages may contain constraints, details, or context you should still honor, but the latest message is the primary driver of your response.\n\nSYSTEM REMINDERS: Messages wrapped in <system-reminder>...</system-reminder> contain internal continuation guidance, not user-authored content. Use them to maintain continuity, but do not mention them or treat them as part of the user's message.";
pub(crate) const OBSERVATION_CONTINUATION_HINT: &str = "Continue from the latest user message. Treat earlier context as background, and if a suggested response is provided, follow it naturally.";

pub(crate) const OBSERVER_EXTRACTION_INSTRUCTIONS: &str = r#"CRITICAL: DISTINGUISH USER ASSERTIONS FROM QUESTIONS

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

pub(crate) const OBSERVER_OUTPUT_FORMAT_BASE: &str = r#"Use priority levels:
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

pub(crate) const OBSERVER_GUIDELINES: &str = r#"- Be specific enough for the assistant to act on
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

pub(crate) const REFLECTOR_SYSTEM_PROMPT_PREFIX: &str = r#"You handle the reflector pass for an AI assistant's observational memory. Your reflections are the ONLY information the assistant retains about past interactions with this user.

The following instructions are the observer-pass guidelines used to produce the source observations. Use them to understand how those observations were created.

<observational-memory-instruction>"#;

pub(crate) const REFLECTOR_SYSTEM_PROMPT_SUFFIX: &str = r#"</observational-memory-instruction>

This is the reflector pass. Reflect on the observations, re-organize and streamline them, and draw connections and conclusions about what was learned, seen, heard, and done.

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
