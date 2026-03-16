use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use serde_json::json;

use crate::agent::PromptSectionName;
use crate::instructions::InstructionSource;
use crate::search::{SearchDoc, SearchMode, limit_from_args, rank_docs, truncate_preview};
use crate::store::{HistoryTurnRecord, Store};
use crate::tools::{UpdatePlanTool, run_blocking};
use crate::{ToolDefinition, ToolParam};

use super::*;

fn agent_id(args: &serde_json::Value) -> String {
    args.get("__agent_id__")
        .and_then(|v| v.as_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("root")
        .to_string()
}

fn build_prompt_environment_context() -> String {
    let mut parts = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        parts.push(format!("Working directory: {}", cwd.display()));

        if cwd.join(".git").exists() {
            parts.push("Git repository: yes".to_string());
        }

        if let Ok(entries) = std::fs::read_dir(&cwd) {
            let mut names: Vec<String> = entries
                .filter_map(|entry| entry.ok())
                .map(|entry| {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if entry.file_type().map(|ty| ty.is_dir()).unwrap_or(false) {
                        format!("{name}/")
                    } else {
                        name
                    }
                })
                .filter(|name| !name.starts_with('.'))
                .collect();
            names.sort();
            if !names.is_empty() {
                parts.push(format!("Top-level entries: {}", names.join(", ")));
            }
        }
    }

    parts.push("REPL third-party packages: none".to_string());
    parts.join("\n")
}

const MIN_RECENT_USER_TURNS: usize = 3;

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
struct HistoryPluginState {
    has_archived_history: bool,
}

fn leading_system_prefix_len(msgs: &[crate::Message]) -> usize {
    msgs.iter()
        .take_while(|msg| msg.role == crate::MessageRole::System)
        .count()
}

fn keep_from_for_recent_turns(msgs: &[crate::Message], prefix_len: usize) -> usize {
    let mut user_turns = 0usize;
    for i in (prefix_len..msgs.len()).rev() {
        if msgs[i].role == crate::MessageRole::User {
            user_turns += 1;
            if user_turns >= MIN_RECENT_USER_TURNS {
                return i;
            }
        }
    }
    prefix_len
}

fn compact_history_messages(
    msgs: &mut Vec<crate::Message>,
    last_context_budget_tokens: usize,
    max_context: usize,
    policy: crate::ContextFoldingConfig,
) -> bool {
    if last_context_budget_tokens == 0 || msgs.is_empty() {
        return false;
    }

    let hard_budget = max_context * usize::from(policy.hard_limit_pct) / 100;
    if last_context_budget_tokens < hard_budget {
        return false;
    }

    let soft_budget = max_context * usize::from(policy.soft_limit_pct) / 100;
    let prefix_len = leading_system_prefix_len(msgs);
    let total_chars: usize = msgs.iter().map(crate::Message::char_count).sum();
    let target_chars = total_chars.saturating_mul(soft_budget) / last_context_budget_tokens.max(1);

    let mut keep_from = msgs.len();
    let mut tail_chars = 0usize;
    for i in (prefix_len..msgs.len()).rev() {
        let cost = msgs[i].char_count();
        if tail_chars + cost > target_chars {
            break;
        }
        tail_chars += cost;
        keep_from = i;
    }

    keep_from = keep_from.min(keep_from_for_recent_turns(msgs, prefix_len));
    if keep_from <= prefix_len {
        return false;
    }

    msgs.drain(prefix_len..keep_from);
    true
}

fn history_message_text(msg: &crate::Message) -> String {
    msg.parts
        .iter()
        .filter_map(|part| match part.kind {
            crate::PartKind::Text | crate::PartKind::Prose | crate::PartKind::Code => {
                Some(part.content.as_str())
            }
            _ => None,
        })
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn latest_turn_history_payload(turn: &crate::AssembledTurn) -> serde_json::Value {
    let messages = &turn.state.messages;
    let turn_index = messages
        .iter()
        .filter(|msg| matches!(msg.role, crate::MessageRole::User))
        .count() as i64;
    let last_user_idx = messages
        .iter()
        .rposition(|msg| matches!(msg.role, crate::MessageRole::User));

    let user_message = last_user_idx
        .and_then(|idx| messages.get(idx))
        .map(history_message_text)
        .unwrap_or_default();

    let mut prose_parts = Vec::new();
    let mut code_parts = Vec::new();
    if let Some(idx) = last_user_idx {
        for msg in messages.iter().skip(idx + 1) {
            if !matches!(msg.role, crate::MessageRole::Assistant) {
                continue;
            }
            for part in &msg.parts {
                match part.kind {
                    crate::PartKind::Text | crate::PartKind::Prose => {
                        if !part.content.trim().is_empty() {
                            prose_parts.push(part.content.clone());
                        }
                    }
                    crate::PartKind::Code => {
                        if !part.content.trim().is_empty() {
                            code_parts.push(part.content.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    let prose = if prose_parts.is_empty() {
        turn.assistant_output.raw_text.clone()
    } else {
        prose_parts.join("\n\n")
    };
    let code = code_parts.join("\n\n");
    let output = turn
        .code_outputs
        .iter()
        .map(|record| match (&record.output, &record.error) {
            (output, Some(error)) if !output.is_empty() && !error.is_empty() => {
                format!("{output}\n{error}")
            }
            (output, _) if !output.is_empty() => output.clone(),
            (_, Some(error)) => error.clone(),
            _ => String::new(),
        })
        .filter(|chunk| !chunk.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    let error = turn.errors.first().map(|issue| issue.message.clone());

    serde_json::json!({
        "index": turn_index,
        "user_message": user_message,
        "prose": prose,
        "code": code,
        "output": output,
        "error": error,
        "tool_calls": turn.tool_calls,
    })
}

fn plan_mode_guidance_message() -> PluginMessage {
    PluginMessage {
        role: crate::MessageRole::System,
        content: r#"
Plan Mode (Conversational)

You work in 3 phases, and you should chat your way to a great plan before finalizing it.
A great plan is detailed enough to hand to another engineer or agent for immediate implementation.
It must be decision complete: the implementer should not need to make important product or technical decisions.

Mode rules (strict)

You are in Plan Mode until a developer message explicitly ends it.
Plan Mode is not changed by user intent, tone, or imperative language.
If the user asks for execution while Plan Mode is active, treat that as a request to plan the execution, not perform it.

Plan Mode vs `update_plan`

Plan Mode is a collaboration mode that can involve user questions and can eventually produce a `<proposed_plan>` block.
Separately, `update_plan` is a checklist/progress/TODO tool for normal execution turns.
Do not use `update_plan` in Plan Mode.
Do not answer with a meta explanation about Plan Mode vs `update_plan` unless the user explicitly asks.

Execution vs mutation in Plan Mode

You may explore and execute non-mutating actions that improve the plan.
You must not perform mutating actions.

Allowed work is exploration that gathers truth, reduces ambiguity, or validates feasibility without changing repo-tracked state.
Examples:
- reading or searching files, configs, schemas, types, manifests, and docs
- static analysis, inspection, and repo exploration
- dry-run style commands when they do not edit repo-tracked files
- tests, builds, or checks that may write caches or build artifacts so long as they do not edit repo-tracked files

Not allowed:
- editing or writing files
- running tools that rewrite repo-tracked files
- applying patches, migrations, or codegen that updates repo-tracked files
- side-effectful commands whose purpose is to carry out the plan rather than refine it

When in doubt: if the action would reasonably be described as doing the work rather than planning the work, do not do it.

PHASE 1 - Ground in the environment

Begin by grounding yourself in the actual environment.
Resolve questions that can be answered through exploration or inspection.
Before asking the user any question, perform at least one targeted non-mutating exploration pass unless the prompt itself has an obvious ambiguity or contradiction.
Do not ask questions that can be answered from the repo or system.

PHASE 2 - Intent chat

Keep asking until you can clearly state:
- goal and success criteria
- audience
- in-scope and out-of-scope work
- constraints
- current state
- the key preferences and tradeoffs

If any high-impact ambiguity remains, do not finalize the plan yet.

PHASE 3 - Implementation chat

Once intent is stable, keep asking until the spec is decision complete:
- approach
- interfaces, APIs, schemas, and I/O shape
- data flow
- edge cases and failure modes
- testing and acceptance criteria
- rollout, compatibility, or migration concerns when materially relevant

Asking questions

Strongly prefer using `ask(...)` for questions that materially change the plan, confirm important assumptions, or request information that cannot be discovered via non-mutating exploration.
Ask questions early for preferences and tradeoffs that are not discoverable from the environment.
If the user does not answer a preference question, proceed with a recommended default and record it as an assumption in the final plan.

Finalization rule

Only output the final plan when it is decision complete and leaves no important decisions to the implementer.
When you present the official plan, wrap it in exactly one `<proposed_plan>` block so the client can render it specially.

`<proposed_plan>` formatting rules:
1. The opening tag must be on its own line.
2. Start the plan content on the next line.
3. The closing tag must be on its own line.
4. Use Markdown inside the block.
5. Keep the tags exactly as `<proposed_plan>` and `</proposed_plan>`.

The final `<proposed_plan>` should be plan-only, concise by default, and implementation-ready.
Prefer a compact structure with:
- a clear title
- a brief summary
- key implementation or interface changes
- tests or acceptance checks
- explicit assumptions and defaults

Only produce at most one `<proposed_plan>` block per turn, and only when you are presenting a complete replacement plan.
"#
        .trim()
        .to_string(),
    }
}

#[derive(Clone, Debug)]
pub struct PlanModePluginConfig {
    pub blocked_tools: BTreeSet<String>,
}

impl Default for PlanModePluginConfig {
    fn default() -> Self {
        Self {
            blocked_tools: [
                "agent_call",
                "agent_kill",
                "agent_result",
                "apply_patch",
                "exec_command",
                "update_plan",
                "write_stdin",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
        }
    }
}

impl PlanModePluginConfig {
    pub fn with_blocked_tools<I, S>(mut self, blocked_tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.blocked_tools = blocked_tools.into_iter().map(Into::into).collect();
        self
    }
}

fn plan_mode_tool_blocked(
    config: &PlanModePluginConfig,
    tool_name: &str,
    _args: &serde_json::Value,
) -> bool {
    config.blocked_tools.contains(tool_name)
}

fn plan_mode_tool_discoverable(config: &PlanModePluginConfig, tool_name: &str) -> bool {
    !config.blocked_tools.contains(tool_name)
}

const PROPOSED_PLAN_OPEN: &str = "<proposed_plan>";
const PROPOSED_PLAN_CLOSE: &str = "</proposed_plan>";
const PLAN_MODE_BADGE_KEY: &str = "mode";
const PLAN_MODE_BADGE_LABEL: &str = "plan";
const PROPOSED_PLAN_TITLE: &str = "PROPOSED PLAN";

#[derive(Debug, Default)]
struct ProposedPlanParser {
    in_tag: bool,
    pending_tag_fragment: String,
    panel_content: String,
}

#[derive(Debug, Default)]
struct ProposedPlanChunk {
    visible_text: String,
}

impl ProposedPlanParser {
    fn push_chunk(&mut self, chunk: &str) -> ProposedPlanChunk {
        let input = format!("{}{}", self.pending_tag_fragment, chunk);
        self.pending_tag_fragment.clear();

        let mut visible_text = String::new();
        let mut idx = 0usize;

        while idx < input.len() {
            let rest = &input[idx..];
            let active_tag = if self.in_tag {
                PROPOSED_PLAN_CLOSE
            } else {
                PROPOSED_PLAN_OPEN
            };

            if rest.starts_with(active_tag) {
                self.in_tag = !self.in_tag;
                idx += active_tag.len();
                continue;
            }

            if active_tag.starts_with(rest)
                || (!self.in_tag && PROPOSED_PLAN_CLOSE.starts_with(rest))
            {
                self.pending_tag_fragment = rest.to_string();
                break;
            }

            if !self.in_tag && rest.starts_with(PROPOSED_PLAN_CLOSE) {
                idx += PROPOSED_PLAN_CLOSE.len();
                continue;
            }

            let Some(ch) = rest.chars().next() else { break };
            if self.in_tag {
                self.panel_content.push(ch);
            } else {
                visible_text.push(ch);
            }
            idx += ch.len_utf8();
        }

        ProposedPlanChunk { visible_text }
    }
}

#[derive(Debug, Default)]
struct PlanModeTurnState {
    panel_key: String,
    parser: ProposedPlanParser,
    emitted_panel_content: String,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
struct PlanModeSnapshot {
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    generation: u64,
    #[serde(default)]
    panel_seq: u64,
}

#[derive(Debug, Default)]
struct PlanModeState {
    enabled: bool,
    generation: u64,
    panel_seq: u64,
    active_turn_applied_generation: Option<u64>,
    active_turn: Option<PlanModeTurnState>,
}

impl PlanModeState {
    fn snapshot(&self) -> PlanModeSnapshot {
        PlanModeSnapshot {
            enabled: self.enabled,
            generation: self.generation,
            panel_seq: self.panel_seq,
        }
    }

    fn set_enabled(&mut self, enabled: bool) -> PlanModeSnapshot {
        if self.enabled != enabled {
            self.enabled = enabled;
            self.generation = self.generation.wrapping_add(1).max(1);
            if !enabled {
                self.active_turn = None;
            }
        }
        self.snapshot()
    }

    fn toggle(&mut self) -> PlanModeSnapshot {
        self.set_enabled(!self.enabled)
    }

    fn prepare_turn(&mut self) -> bool {
        self.active_turn_applied_generation = None;
        self.active_turn = None;
        if !self.enabled {
            return false;
        }
        self.panel_seq = self.panel_seq.wrapping_add(1).max(1);
        self.active_turn = Some(PlanModeTurnState {
            panel_key: format!("proposed_plan:{}", self.panel_seq),
            parser: ProposedPlanParser::default(),
            emitted_panel_content: String::new(),
        });
        self.active_turn_applied_generation = Some(self.generation);
        true
    }

    fn checkpoint_injection_needed(&mut self) -> bool {
        if !self.enabled || self.active_turn_applied_generation == Some(self.generation) {
            return false;
        }
        self.active_turn_applied_generation = Some(self.generation);
        true
    }

    fn finish_turn(&mut self) {
        self.active_turn_applied_generation = None;
        self.active_turn = None;
    }

    fn badge_event(&self) -> crate::plugin::PluginSurfaceEvent {
        crate::plugin::PluginSurfaceEvent::ModeIndicatorUpsert {
            key: PLAN_MODE_BADGE_KEY.to_string(),
            label: PLAN_MODE_BADGE_LABEL.to_string(),
        }
    }

    fn transform_stream_chunk(
        &mut self,
        chunk: String,
    ) -> (String, Vec<crate::plugin::PluginSurfaceEvent>) {
        let Some(turn) = self.active_turn.as_mut() else {
            return (chunk, Vec::new());
        };
        let parsed = turn.parser.push_chunk(&chunk);
        let mut events = Vec::new();
        let panel_content = turn.parser.panel_content.trim().to_string();
        if !panel_content.is_empty() && panel_content != turn.emitted_panel_content {
            turn.emitted_panel_content = panel_content.clone();
            events.push(crate::plugin::PluginSurfaceEvent::PanelUpsert {
                key: turn.panel_key.clone(),
                title: PROPOSED_PLAN_TITLE.to_string(),
                content: panel_content,
            });
        }
        (parsed.visible_text, events)
    }

    fn transform_response(
        &mut self,
        response: crate::llm::types::LlmResponse,
    ) -> (
        crate::llm::types::LlmResponse,
        Vec<crate::plugin::PluginSurfaceEvent>,
    ) {
        if !self.enabled {
            return (response, Vec::new());
        }

        let mut parser = ProposedPlanParser::default();
        let mut sanitized_parts = Vec::new();
        let source_parts = if response.parts.is_empty() && !response.full_text.is_empty() {
            vec![crate::llm::types::LlmOutputPart::Text {
                text: response.full_text.clone(),
            }]
        } else {
            response.parts.clone()
        };

        let mut sanitized_deltas = Vec::new();
        let mut sanitized_full_text = String::new();
        for part in source_parts {
            match part {
                crate::llm::types::LlmOutputPart::Text { text } => {
                    let parsed = parser.push_chunk(&text);
                    if !parsed.visible_text.is_empty() {
                        sanitized_full_text.push_str(&parsed.visible_text);
                        sanitized_deltas.push(parsed.visible_text.clone());
                        sanitized_parts.push(crate::llm::types::LlmOutputPart::Text {
                            text: parsed.visible_text,
                        });
                    }
                }
                other => sanitized_parts.push(other),
            }
        }

        let mut events = Vec::new();
        if let Some(turn) = self.active_turn.as_mut() {
            let panel_content = parser.panel_content.trim().to_string();
            if !panel_content.is_empty() && panel_content != turn.emitted_panel_content {
                turn.emitted_panel_content = panel_content.clone();
                events.push(crate::plugin::PluginSurfaceEvent::PanelUpsert {
                    key: turn.panel_key.clone(),
                    title: PROPOSED_PLAN_TITLE.to_string(),
                    content: panel_content,
                });
            }
        }

        (
            crate::llm::types::LlmResponse {
                full_text: sanitized_full_text,
                deltas: sanitized_deltas,
                parts: sanitized_parts,
                usage: response.usage,
                request_body: response.request_body,
                http_summary: response.http_summary,
            },
            events,
        )
    }

    fn restore_snapshot(&mut self, snapshot: PlanModeSnapshot) {
        self.enabled = snapshot.enabled;
        self.generation = snapshot.generation;
        self.panel_seq = snapshot.panel_seq;
        self.active_turn_applied_generation = None;
        self.active_turn = None;
    }
}

#[derive(Clone)]
struct HistoryTools {
    store: Arc<Store>,
}

impl HistoryTools {
    fn new(store: Arc<Store>) -> Self {
        Self { store }
    }

    fn history_item(turn: &HistoryTurnRecord) -> serde_json::Value {
        let preview_source = if !turn.user_message.trim().is_empty() {
            &turn.user_message
        } else if !turn.prose.trim().is_empty() {
            &turn.prose
        } else {
            &turn.output
        };
        json!({
            "turn": turn.index,
            "preview": truncate_preview(preview_source, 220),
            "tool_calls": turn.tool_calls,
            "files_read": turn.files_read,
            "files_written": turn.files_written,
        })
    }

    fn search_history(&self, args: &serde_json::Value) -> ToolResult {
        let agent_id = agent_id(args);
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let browse_all = query.trim().is_empty();
        let mode = SearchMode::parse(args.get("mode").and_then(|v| v.as_str()));
        let regex = args.get("regex").and_then(|v| v.as_str());
        let limit = if browse_all && args.get("limit").is_none() {
            usize::MAX
        } else {
            limit_from_args(args)
        };
        let since_turn = args.get("since_turn").and_then(|v| v.as_i64());

        let selected_fields: HashSet<String> = args
            .get("fields")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(str::to_string)
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_else(|| {
                [
                    "user_message".to_string(),
                    "code".to_string(),
                    "prose".to_string(),
                    "output".to_string(),
                    "tool_calls".to_string(),
                ]
                .into_iter()
                .collect()
            });

        let turns: Vec<HistoryTurnRecord> = self
            .store
            .history_export(&agent_id)
            .into_iter()
            .filter(|turn| since_turn.is_none_or(|min_turn| turn.index >= min_turn))
            .collect();

        if browse_all {
            return ToolResult::ok(json!(
                turns
                    .iter()
                    .take(limit)
                    .map(Self::history_item)
                    .collect::<Vec<_>>()
            ));
        }

        let docs: Vec<SearchDoc> = turns
            .iter()
            .map(|turn| {
                let mut fields = HashMap::new();
                if selected_fields.contains("user_message") {
                    fields.insert("user_message", turn.user_message.clone());
                }
                if selected_fields.contains("code") {
                    fields.insert("code", turn.code.clone());
                }
                if selected_fields.contains("prose") {
                    fields.insert("prose", turn.prose.clone());
                }
                if selected_fields.contains("output") {
                    fields.insert("output", turn.output.clone());
                }
                if selected_fields.contains("tool_calls") {
                    let tool_calls = serde_json::to_string(&turn.tool_calls).unwrap_or_default();
                    fields.insert("tool_calls", tool_calls);
                }
                SearchDoc { fields }
            })
            .collect();

        let ranked = rank_docs(
            &docs,
            &query,
            mode,
            regex,
            &[
                ("user_message", 3.0),
                ("prose", 2.0),
                ("code", 2.0),
                ("output", 1.0),
                ("tool_calls", 1.0),
            ],
        );

        let items: Vec<serde_json::Value> = ranked
            .into_iter()
            .take(limit)
            .map(|(idx, score, hits)| {
                let turn = &turns[idx];
                let mut item = Self::history_item(turn);
                if let Some(obj) = item.as_object_mut() {
                    obj.insert("score".to_string(), json!(score));
                    obj.insert("field_hits".to_string(), json!(hits));
                }
                item
            })
            .collect();
        ToolResult::ok(json!(items))
    }
}

#[async_trait::async_trait]
impl ToolProvider for HistoryTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "search_history".into(),
            description: "Search turn history using hybrid/literal/regex matching. With no `query`, returns the full turn list in stable turn order.".into(),
            params: vec![
                ToolParam::optional("query", "str"),
                ToolParam::optional("mode", "str"),
                ToolParam::optional("regex", "str"),
                ToolParam::optional("limit", "int"),
                ToolParam::optional("fields", "list"),
                ToolParam::optional("since_turn", "int"),
            ],
            returns: "list".into(),
            examples: vec![],
            enabled: false,
            injected: false,
        }]
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        let tools = self.clone();
        let name = name.to_string();
        let args = args.clone();
        run_blocking(move || match name.as_str() {
            "search_history" => tools.search_history(&args),
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        })
        .await
    }
}

fn history_prompt_contributions(context: &crate::PromptContext) -> Vec<crate::PromptContribution> {
    if !context.has_tool("search_history") {
        return Vec::new();
    }

    vec![crate::PromptContribution::guidance(
        "### History\nUse `search_history` only when older context actually matters. With no query, it lists archived turns in stable turn order; add a focused query when you need retrieval instead of browsing.",
    )]
}

pub struct HistoryPluginFactory {
    store: Arc<Store>,
    tools: Arc<HistoryTools>,
}

impl HistoryPluginFactory {
    pub fn new(store: Arc<Store>) -> Self {
        Self {
            tools: Arc::new(HistoryTools::new(Arc::clone(&store))),
            store,
        }
    }
}

impl PluginFactory for HistoryPluginFactory {
    fn id(&self) -> &'static str {
        "history"
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(HistoryPlugin {
            store: Arc::clone(&self.store),
            tools: Arc::clone(&self.tools),
            agent_id: ctx.agent_id.clone(),
            state: Arc::new(Mutex::new(HistoryPluginState::default())),
        }))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PromptContextPluginConfig {
    pub include_environment: bool,
    pub include_project_instructions: bool,
}

impl Default for PromptContextPluginConfig {
    fn default() -> Self {
        Self {
            include_environment: true,
            include_project_instructions: true,
        }
    }
}

pub struct PromptContextPluginFactory {
    instruction_source: Arc<dyn InstructionSource>,
    config: PromptContextPluginConfig,
}

impl PromptContextPluginFactory {
    pub fn new(
        instruction_source: Arc<dyn InstructionSource>,
        config: PromptContextPluginConfig,
    ) -> Self {
        Self {
            instruction_source,
            config,
        }
    }
}

impl PluginFactory for PromptContextPluginFactory {
    fn id(&self) -> &'static str {
        "prompt_context"
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(PromptContextPlugin {
            instruction_source: Arc::clone(&self.instruction_source),
            config: self.config.clone(),
        }))
    }
}

struct PromptContextPlugin {
    instruction_source: Arc<dyn InstructionSource>,
    config: PromptContextPluginConfig,
}

impl SessionPlugin for PromptContextPlugin {
    fn id(&self) -> &'static str {
        "prompt_context"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        let instruction_source = Arc::clone(&self.instruction_source);
        let config = self.config.clone();
        reg.prompt().contribute(Arc::new(move |_ctx| {
            let instruction_source = Arc::clone(&instruction_source);
            let config = config.clone();
            Box::pin(async move {
                let mut contributions = Vec::new();
                let base_context = build_prompt_environment_context();
                if config.include_environment && !base_context.trim().is_empty() {
                    contributions.push(PromptContribution {
                        section: PromptSectionName::Environment,
                        priority: 0,
                        content: base_context,
                    });
                }
                let project_instructions = instruction_source.system_instructions();
                if config.include_project_instructions && !project_instructions.trim().is_empty() {
                    contributions.push(PromptContribution {
                        section: PromptSectionName::Guidance,
                        priority: 0,
                        content: format!("### Project Instructions\n{}", project_instructions),
                    });
                }
                Ok(contributions)
            })
        }));
        Ok(())
    }
}

struct HistoryPlugin {
    store: Arc<Store>,
    tools: Arc<HistoryTools>,
    agent_id: String,
    state: Arc<Mutex<HistoryPluginState>>,
}

impl SessionPlugin for HistoryPlugin {
    fn id(&self) -> &'static str {
        "history"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.tools()
            .provider(Arc::clone(&self.tools) as Arc<dyn ToolProvider>)?;
        reg.prompt().contribute(Arc::new(move |ctx| {
            Box::pin(async move { Ok(history_prompt_contributions(&ctx.prompt)) })
        }));
        let state = Arc::clone(&self.state);
        reg.messages().mutator(
            MessageMutatorHook::AfterTokenCount,
            Arc::new(move |ctx, mut messages| {
                let state = Arc::clone(&state);
                Box::pin(async move {
                    let Some(prompt_usage) = ctx.prompt_usage else {
                        return Ok(messages);
                    };
                    let Some(max_context) = ctx.max_context_tokens else {
                        return Ok(messages);
                    };
                    let policy = ctx.context_folding.unwrap_or_default();
                    if compact_history_messages(
                        &mut messages,
                        prompt_usage.context_budget_tokens,
                        max_context,
                        policy,
                    ) {
                        state
                            .lock()
                            .map_err(|_| {
                                PluginError::Session("history plugin state poisoned".to_string())
                            })?
                            .has_archived_history = true;
                    }
                    Ok(messages)
                })
            }),
        )?;
        let state = Arc::clone(&self.state);
        reg.surface().contribute(Arc::new(move |ctx| {
            let has_archived_history = state
                .lock()
                .map_err(|_| PluginError::Session("history plugin state poisoned".to_string()))?
                .has_archived_history;
            if !ctx.tools.iter().any(|tool| tool.name == "search_history") {
                return Ok(ToolSurfaceContribution::default());
            }
            Ok(ToolSurfaceContribution {
                overrides: vec![ToolSurfaceOverride {
                    tool_name: "search_history".to_string(),
                    enabled: Some(has_archived_history),
                    injected: Some(has_archived_history),
                }],
                tool_list_notes: Vec::new(),
            })
        }));
        let state = Arc::clone(&self.state);
        reg.prompt().contribute(Arc::new(move |ctx| {
            let state = Arc::clone(&state);
            Box::pin(async move {
                let has_archived_history = state
                    .lock()
                    .map_err(|_| PluginError::Session("history plugin state poisoned".to_string()))?
                    .has_archived_history;
                let mut contributions = history_prompt_contributions(&ctx.prompt);
                if has_archived_history {
                    contributions.push(PromptContribution::guidance(
                        "Older turns were archived outside the active context. Use `search_history(...)` only when older context actually matters.",
                    ));
                }
                Ok(contributions)
            })
        }));

        let store = Arc::clone(&self.store);
        let agent_id = self.agent_id.clone();
        reg.turn().committed(Arc::new(move |turn| {
            let store = Arc::clone(&store);
            let agent_id = agent_id.clone();
            Box::pin(async move {
                store.history_add_turn(&agent_id, &latest_turn_history_payload(&turn));
                Ok(())
            })
        }));

        let plugin_state = Arc::clone(&self.state);
        reg.session()
            .config_mutator(Arc::new(move |ctx, mut state| {
                let plugin_state = Arc::clone(&plugin_state);
                Box::pin(async move {
                    let Some(max_context) = ctx.current.context_window else {
                        return Ok(state);
                    };
                    let Some(prompt_usage) = state.last_prompt_usage.clone() else {
                        return Ok(state);
                    };
                    if ctx.previous.context_window == Some(max_context)
                        && ctx.previous.model == ctx.current.model
                        && ctx.previous.provider_kind == ctx.current.provider_kind
                    {
                        return Ok(state);
                    }
                    if compact_history_messages(
                        &mut state.messages,
                        prompt_usage.context_budget_tokens,
                        max_context as usize,
                        state.context_folding,
                    ) {
                        plugin_state
                            .lock()
                            .map_err(|_| {
                                PluginError::Session("history plugin state poisoned".to_string())
                            })?
                            .has_archived_history = true;
                    }
                    Ok(state)
                })
            }));

        let store = Arc::clone(&self.store);
        reg.external().op(
            ExternalOpDef {
                name: "history.summary".to_string(),
                description: "Summarize the recent turn history for a session.".to_string(),
                kind: ExternalOpKind::Query,
                session_param: SessionParam::Required,
                input_schema: json!({
                    "type":"object",
                    "properties":{"limit":{"type":"integer","minimum":1}},
                    "additionalProperties": false
                }),
                output_schema: json!({
                    "type":"object",
                    "properties":{
                        "session_id":{"type":"string"},
                        "turn_count":{"type":"integer"},
                        "latest_user_message":{"type":["string","null"]},
                        "recent_turns":{"type":"array"}
                    },
                    "required":["session_id","turn_count","recent_turns"],
                    "additionalProperties": false
                }),
            },
            Arc::new(move |ctx, args| {
                let store = Arc::clone(&store);
                Box::pin(async move {
                    let Some(session_id) = ctx.session_id else {
                        return ToolResult::err(json!("history.summary requires session_id"));
                    };
                    let limit = args
                        .get("limit")
                        .and_then(|v| v.as_u64())
                        .and_then(|v| usize::try_from(v).ok())
                        .unwrap_or(5)
                        .clamp(1, 20);
                    let turns = store.history_export(&session_id);
                    let latest_user_message = turns.iter().rev().find_map(|turn| {
                        (!turn.user_message.trim().is_empty()).then_some(turn.user_message.clone())
                    });
                    let recent_turns = turns
                        .iter()
                        .rev()
                        .take(limit)
                        .map(|turn| {
                            let preview_source = if !turn.user_message.trim().is_empty() {
                                &turn.user_message
                            } else if !turn.prose.trim().is_empty() {
                                &turn.prose
                            } else {
                                &turn.output
                            };
                            json!({
                                "turn": turn.index,
                                "preview": truncate_preview(preview_source, 180),
                                "tool_calls": turn.tool_calls.len(),
                            })
                        })
                        .collect::<Vec<_>>();
                    ToolResult::ok(json!({
                        "session_id": session_id,
                        "turn_count": turns.len(),
                        "latest_user_message": latest_user_message,
                        "recent_turns": recent_turns,
                    }))
                })
            }),
        )?;
        Ok(())
    }

    fn snapshot(
        &self,
        _writer: &mut dyn SnapshotWriter,
    ) -> Result<PluginSnapshotMeta, PluginError> {
        let turns = self
            .store
            .history_export(&self.agent_id)
            .into_iter()
            .map(|turn| {
                json!({
                    "index": turn.index,
                    "user_message": turn.user_message,
                    "prose": turn.prose,
                    "code": turn.code,
                    "output": turn.output,
                    "error": turn.error,
                    "tool_calls": turn.tool_calls,
                    "files_read": turn.files_read,
                    "files_written": turn.files_written,
                })
            })
            .collect::<Vec<_>>();
        let state = self
            .state
            .lock()
            .map_err(|_| PluginError::Snapshot("history plugin state poisoned".to_string()))?;
        Ok(PluginSnapshotMeta {
            plugin_id: self.id().to_string(),
            plugin_version: self.version().to_string(),
            state: Some(json!({
                "turns": turns,
                "has_archived_history": state.has_archived_history,
            })),
        })
    }

    fn restore(
        &self,
        meta: &PluginSnapshotMeta,
        _reader: &dyn SnapshotReader,
    ) -> Result<(), PluginError> {
        let turns = meta
            .state
            .as_ref()
            .and_then(|state| state.get("turns"))
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap_or_default();
        self.store.history_load(&self.agent_id, &turns);
        let has_archived_history = meta
            .state
            .as_ref()
            .and_then(|state| state.get("has_archived_history"))
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        self.state
            .lock()
            .map_err(|_| PluginError::Snapshot("history plugin state poisoned".to_string()))?
            .has_archived_history = has_archived_history;
        Ok(())
    }
}

pub struct PlanTrackerPluginFactory;

impl PluginFactory for PlanTrackerPluginFactory {
    fn id(&self) -> &'static str {
        "plan_tracker"
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(PlanTrackerPlugin {
            tools: Arc::new(UpdatePlanTool::default()),
        }))
    }
}

struct PlanTrackerPlugin {
    tools: Arc<UpdatePlanTool>,
}

impl SessionPlugin for PlanTrackerPlugin {
    fn id(&self) -> &'static str {
        "plan_tracker"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.tools()
            .provider(Arc::clone(&self.tools) as Arc<dyn ToolProvider>)
    }

    fn snapshot(
        &self,
        _writer: &mut dyn SnapshotWriter,
    ) -> Result<PluginSnapshotMeta, PluginError> {
        let snapshot = self
            .tools
            .snapshot()
            .map_err(|err| PluginError::Snapshot(err.to_string()))?;
        Ok(PluginSnapshotMeta {
            plugin_id: self.id().to_string(),
            plugin_version: self.version().to_string(),
            state: Some(
                serde_json::to_value(snapshot)
                    .map_err(|err| PluginError::Snapshot(err.to_string()))?,
            ),
        })
    }

    fn restore(
        &self,
        meta: &PluginSnapshotMeta,
        _reader: &dyn SnapshotReader,
    ) -> Result<(), PluginError> {
        let snapshot = meta
            .state
            .clone()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|err| PluginError::Snapshot(err.to_string()))?
            .unwrap_or_default();
        self.tools
            .restore(snapshot)
            .map_err(PluginError::Snapshot)?;
        Ok(())
    }
}

pub struct PlanModePluginFactory {
    config: PlanModePluginConfig,
}

impl Default for PlanModePluginFactory {
    fn default() -> Self {
        Self::new(PlanModePluginConfig::default())
    }
}

impl PlanModePluginFactory {
    pub fn new(config: PlanModePluginConfig) -> Self {
        Self { config }
    }
}

impl PluginFactory for PlanModePluginFactory {
    fn id(&self) -> &'static str {
        "plan_mode"
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(PlanModePlugin {
            state: Arc::new(Mutex::new(PlanModeState::default())),
            config: self.config.clone(),
        }))
    }
}

struct PlanModePlugin {
    state: Arc<Mutex<PlanModeState>>,
    config: PlanModePluginConfig,
}

impl SessionPlugin for PlanModePlugin {
    fn id(&self) -> &'static str {
        "plan_mode"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        let before_turn_state = Arc::clone(&self.state);
        reg.turn().before(Arc::new(move |_ctx| {
            let state = Arc::clone(&before_turn_state);
            Box::pin(async move {
                let mut state = state
                    .lock()
                    .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?;
                let should_inject = state.prepare_turn();
                Ok(if should_inject {
                    vec![
                        PluginDirective::emit_events(vec![state.badge_event()]),
                        PluginDirective::EnqueueMessages {
                            messages: vec![plan_mode_guidance_message()],
                        },
                    ]
                } else {
                    Vec::new()
                })
            })
        }));

        let checkpoint_state = Arc::clone(&self.state);
        reg.turn().checkpoint(Arc::new(move |_ctx| {
            let state = Arc::clone(&checkpoint_state);
            Box::pin(async move {
                let mut state = state
                    .lock()
                    .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?;
                let should_inject = state.checkpoint_injection_needed();
                Ok(if should_inject {
                    vec![
                        PluginDirective::emit_events(vec![state.badge_event()]),
                        PluginDirective::EnqueueMessages {
                            messages: vec![plan_mode_guidance_message()],
                        },
                    ]
                } else {
                    Vec::new()
                })
            })
        }));

        let after_turn_state = Arc::clone(&self.state);
        reg.turn().after(Arc::new(move |_ctx| {
            let state = Arc::clone(&after_turn_state);
            Box::pin(async move {
                state
                    .lock()
                    .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?
                    .finish_turn();
                Ok(Vec::new())
            })
        }));

        let stream_state = Arc::clone(&self.state);
        reg.output().stream(Arc::new(move |ctx| {
            let state = Arc::clone(&stream_state);
            Box::pin(async move {
                let mut state = state
                    .lock()
                    .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?;
                let (chunk, events) = if state.enabled {
                    state.transform_stream_chunk(ctx.chunk)
                } else {
                    (ctx.chunk, Vec::new())
                };
                Ok(crate::plugin::AssistantStreamTransform { chunk, events })
            })
        }));

        let response_state = Arc::clone(&self.state);
        reg.output().response(Arc::new(move |ctx| {
            let state = Arc::clone(&response_state);
            Box::pin(async move {
                let mut state = state
                    .lock()
                    .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?;
                let (response, events) = if state.enabled {
                    state.transform_response(ctx.response)
                } else {
                    (ctx.response, Vec::new())
                };
                Ok(crate::plugin::AssistantResponseTransform { response, events })
            })
        }));

        let before_tool_state = Arc::clone(&self.state);
        let before_tool_config = self.config.clone();
        reg.tool_calls().before(Arc::new(move |ctx| {
            let state = Arc::clone(&before_tool_state);
            let config = before_tool_config.clone();
            Box::pin(async move {
                let enabled = state
                    .lock()
                    .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?
                    .enabled;
                if !enabled {
                    return Ok(Vec::new());
                }
                if !plan_mode_tool_blocked(&config, &ctx.tool_name, &ctx.args) {
                    return Ok(Vec::new());
                }
                Ok(vec![PluginDirective::AbortTurn {
                    code: "plan_mode_tool_blocked".to_string(),
                    message: format!(
                        "Plan mode blocks `{}`. Disable plan mode to execute implementation tools.",
                        ctx.tool_name
                    ),
                }])
            })
        }));

        let tool_surface_state = Arc::clone(&self.state);
        let tool_surface_config = self.config.clone();
        reg.surface().contribute(Arc::new(move |ctx| {
            let enabled = tool_surface_state
                .lock()
                .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?
                .enabled;
            if !enabled {
                return Ok(ToolSurfaceContribution::default());
            }
            let config = tool_surface_config.clone();
            let overrides = ctx
                .tools
                .iter()
                .filter(|tool| !plan_mode_tool_discoverable(&config, &tool.name))
                .map(|tool| ToolSurfaceOverride {
                    tool_name: tool.name.clone(),
                    enabled: Some(false),
                    injected: Some(false),
                })
                .collect();
            Ok(ToolSurfaceContribution {
                overrides,
                tool_list_notes: Vec::new(),
            })
        }));

        let status_state = Arc::clone(&self.state);
        reg.external().op(
            ExternalOpDef {
                name: "plan_mode.status".to_string(),
                description: "Read the current plan-mode state for this session.".to_string(),
                kind: ExternalOpKind::Query,
                session_param: SessionParam::Required,
                input_schema: json!({
                    "type": "object",
                    "additionalProperties": false
                }),
                output_schema: json!({
                    "type": "object",
                    "properties": {
                        "session_id": { "type": "string" },
                        "enabled": { "type": "boolean" }
                    },
                    "required": ["session_id", "enabled"],
                    "additionalProperties": false
                }),
            },
            Arc::new(move |ctx, _args| {
                let state = Arc::clone(&status_state);
                Box::pin(async move {
                    let Some(session_id) = ctx.session_id else {
                        return ToolResult::err(json!("plan_mode.status requires session_id"));
                    };
                    let enabled = match state.lock() {
                        Ok(guard) => guard.enabled,
                        Err(_) => return ToolResult::err(json!("plan mode state poisoned")),
                    };
                    ToolResult::ok(json!({
                        "session_id": session_id,
                        "enabled": enabled,
                    }))
                })
            }),
        )?;

        for (name, description, kind) in [
            (
                "plan_mode.enable",
                "Enable plan mode for this session.",
                ExternalOpKind::Command,
            ),
            (
                "plan_mode.disable",
                "Disable plan mode for this session.",
                ExternalOpKind::Command,
            ),
            (
                "plan_mode.toggle",
                "Toggle plan mode for this session.",
                ExternalOpKind::Command,
            ),
        ] {
            let state = Arc::clone(&self.state);
            reg.external().op(
                ExternalOpDef {
                    name: name.to_string(),
                    description: description.to_string(),
                    kind,
                    session_param: SessionParam::Required,
                    input_schema: json!({
                        "type": "object",
                        "additionalProperties": false
                    }),
                    output_schema: json!({
                        "type": "object",
                        "properties": {
                            "session_id": { "type": "string" },
                            "enabled": { "type": "boolean" }
                        },
                        "required": ["session_id", "enabled"],
                        "additionalProperties": false
                    }),
                },
                Arc::new(move |ctx, _args| {
                    let state = Arc::clone(&state);
                    let op_name = name.to_string();
                    Box::pin(async move {
                        let Some(session_id) = ctx.session_id else {
                            return ToolResult::err(json!(format!(
                                "{op_name} requires session_id"
                            )));
                        };
                        let snapshot = match state.lock() {
                            Ok(mut guard) => match op_name.as_str() {
                                "plan_mode.enable" => guard.set_enabled(true),
                                "plan_mode.disable" => guard.set_enabled(false),
                                "plan_mode.toggle" => guard.toggle(),
                                _ => unreachable!(),
                            },
                            Err(_) => return ToolResult::err(json!("plan mode state poisoned")),
                        };
                        ToolResult::ok(json!({
                            "session_id": session_id,
                            "enabled": snapshot.enabled,
                        }))
                    })
                }),
            )?;
        }

        Ok(())
    }

    fn snapshot(
        &self,
        _writer: &mut dyn SnapshotWriter,
    ) -> Result<PluginSnapshotMeta, PluginError> {
        let snapshot = self
            .state
            .lock()
            .map_err(|_| PluginError::Snapshot("plan mode state poisoned".to_string()))?
            .snapshot();
        Ok(PluginSnapshotMeta {
            plugin_id: self.id().to_string(),
            plugin_version: self.version().to_string(),
            state: Some(json!({
                "enabled": snapshot.enabled,
                "generation": snapshot.generation,
            })),
        })
    }

    fn restore(
        &self,
        meta: &PluginSnapshotMeta,
        _reader: &dyn SnapshotReader,
    ) -> Result<(), PluginError> {
        let snapshot = meta
            .state
            .clone()
            .map(serde_json::from_value::<PlanModeSnapshot>)
            .transpose()
            .map_err(|err| PluginError::Snapshot(err.to_string()))?
            .unwrap_or_default();
        self.state
            .lock()
            .map_err(|_| PluginError::Snapshot("plan mode state poisoned".to_string()))?
            .restore_snapshot(snapshot);
        Ok(())
    }
}

pub use HistoryPluginFactory as BuiltinHistoryPluginFactory;
pub use PlanModePluginFactory as BuiltinPlanModePluginFactory;
pub use PlanTrackerPluginFactory as BuiltinPlanTrackerPluginFactory;
pub use PromptContextPluginFactory as BuiltinPromptContextPluginFactory;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::InstructionSource;
    use crate::plugin::{MessageMutatorContext, MessageMutatorHook, ToolCallHookContext};
    use crate::tools::StateToolsPluginFactory;
    use crate::{
        AgentStateEnvelope, AssembledTurn, AssistantOutput, ContextFoldingConfig, DoneReason,
        ExecutionMode, MessageRole, OutputState, PluginHost, PromptUsage, SessionCreateRequest,
        SessionHandle, SessionManager, SessionSnapshot, TokenUsage, TurnHookContext, TurnInput,
        TurnResultHookContext, TurnStatus,
    };

    struct MockSessionManager;

    struct StaticInstructionSource {
        text: String,
        read_text: String,
    }

    impl InstructionSource for StaticInstructionSource {
        fn system_instructions(&self) -> String {
            self.text.clone()
        }

        fn context_instructions_for_reads(&self, _read_paths: &[String]) -> String {
            self.read_text.clone()
        }
    }

    #[async_trait::async_trait]
    impl SessionManager for MockSessionManager {
        async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
            Ok(AgentStateEnvelope::default())
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<SessionSnapshot, PluginError> {
            Ok(AgentStateEnvelope::default())
        }

        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, PluginError> {
            Ok(Vec::new())
        }

        async fn create_session(
            &self,
            request: SessionCreateRequest,
        ) -> Result<SessionHandle, PluginError> {
            Ok(SessionHandle {
                session_id: request.agent_id.unwrap_or_else(|| "child".to_string()),
                config: crate::SessionConfigSnapshot {
                    provider_kind: crate::provider::ProviderKind::OpenAiGeneric,
                    model: "mock-model".to_string(),
                    model_variant: None,
                    execution_mode: ExecutionMode::Standard,
                    context_folding: ContextFoldingConfig::default(),
                    context_window: None,
                    max_turns: None,
                    include_soul: false,
                    sub_agent: false,
                },
            })
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
            Ok(())
        }

        async fn start_turn_stream(
            &self,
            session_id: &str,
            _input: TurnInput,
        ) -> Result<crate::plugin::SessionTurnHandle, PluginError> {
            let (_tx, rx) = tokio::sync::mpsc::channel(1);
            Ok(crate::plugin::SessionTurnHandle {
                turn_id: format!("{session_id}-turn"),
                session_id: session_id.to_string(),
                config: crate::SessionConfigSnapshot {
                    provider_kind: crate::provider::ProviderKind::OpenAiGeneric,
                    model: "mock-model".to_string(),
                    model_variant: None,
                    execution_mode: ExecutionMode::Standard,
                    context_folding: ContextFoldingConfig::default(),
                    context_window: None,
                    max_turns: None,
                    include_soul: false,
                    sub_agent: false,
                },
                events: rx,
            })
        }

        async fn await_turn(&self, turn_id: &str) -> Result<AssembledTurn, PluginError> {
            Ok(empty_turn(turn_id.trim_end_matches("-turn")))
        }

        async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
            Ok(())
        }
    }

    fn empty_turn(session_id: &str) -> AssembledTurn {
        AssembledTurn {
            state: AgentStateEnvelope {
                agent_id: session_id.to_string(),
                execution_mode: ExecutionMode::Standard,
                context_folding: ContextFoldingConfig::default(),
                ..Default::default()
            },
            status: TurnStatus::Completed,
            assistant_output: AssistantOutput {
                safe_text: String::new(),
                raw_text: String::new(),
                state: OutputState::Usable,
            },
            done_reason: DoneReason::ModelStop,
            execution: crate::ExecutionSummary {
                mode: ExecutionMode::Standard,
                had_tool_calls: false,
                had_code_execution: false,
            },
            token_usage: TokenUsage::default(),
            tool_calls: Vec::new(),
            code_outputs: Vec::new(),
            errors: Vec::new(),
        }
    }

    #[cfg(feature = "sqlite-store")]
    #[tokio::test]
    async fn history_plugin_emits_turn_prompt_note_only_after_compaction() {
        let store = Arc::new(Store::memory().expect("store"));
        let host = PluginHost::new(vec![Arc::new(BuiltinHistoryPluginFactory::new(store))]);
        let session = host.build_standard_session("root", None).expect("session");
        let manager: Arc<dyn SessionManager> = Arc::new(MockSessionManager);

        let empty = session
            .collect_prompt_contributions(PromptHookContext {
                session_id: "root".to_string(),
                host: Arc::clone(&manager),
                prompt: crate::PromptContext {
                    tool_names: vec!["search_history".to_string()],
                    ..Default::default()
                },
                state: AgentStateEnvelope::default(),
            })
            .await
            .expect("prompt");
        assert_eq!(empty.len(), 1);

        let messages = vec![
            crate::Message {
                id: "u1".to_string(),
                role: MessageRole::User,
                parts: vec![crate::Part {
                    id: "u1.p0".to_string(),
                    kind: crate::PartKind::Text,
                    content: "oldest user".repeat(20),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: crate::PruneState::Intact,
                }],
                origin: None,
            },
            crate::Message {
                id: "a1".to_string(),
                role: MessageRole::Assistant,
                parts: vec![crate::Part {
                    id: "a1.p0".to_string(),
                    kind: crate::PartKind::Text,
                    content: "oldest assistant".repeat(20),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: crate::PruneState::Intact,
                }],
                origin: None,
            },
            crate::Message {
                id: "u2".to_string(),
                role: MessageRole::User,
                parts: vec![crate::Part {
                    id: "u2.p0".to_string(),
                    kind: crate::PartKind::Text,
                    content: "older user".repeat(20),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: crate::PruneState::Intact,
                }],
                origin: None,
            },
            crate::Message {
                id: "a2".to_string(),
                role: MessageRole::Assistant,
                parts: vec![crate::Part {
                    id: "a2.p0".to_string(),
                    kind: crate::PartKind::Text,
                    content: "older assistant".repeat(20),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: crate::PruneState::Intact,
                }],
                origin: None,
            },
            crate::Message {
                id: "u3".to_string(),
                role: MessageRole::User,
                parts: vec![crate::Part {
                    id: "u3.p0".to_string(),
                    kind: crate::PartKind::Text,
                    content: "recent user".repeat(20),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: crate::PruneState::Intact,
                }],
                origin: None,
            },
            crate::Message {
                id: "a3".to_string(),
                role: MessageRole::Assistant,
                parts: vec![crate::Part {
                    id: "a3.p0".to_string(),
                    kind: crate::PartKind::Text,
                    content: "recent assistant".repeat(20),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: crate::PruneState::Intact,
                }],
                origin: None,
            },
            crate::Message {
                id: "u4".to_string(),
                role: MessageRole::User,
                parts: vec![crate::Part {
                    id: "u4.p0".to_string(),
                    kind: crate::PartKind::Text,
                    content: "latest user".repeat(20),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: crate::PruneState::Intact,
                }],
                origin: None,
            },
            crate::Message {
                id: "a4".to_string(),
                role: MessageRole::Assistant,
                parts: vec![crate::Part {
                    id: "a4.p0".to_string(),
                    kind: crate::PartKind::Text,
                    content: "latest assistant".repeat(20),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: crate::PruneState::Intact,
                }],
                origin: None,
            },
        ];
        let compacted = session
            .mutate_messages(
                MessageMutatorContext {
                    hook: MessageMutatorHook::AfterTokenCount,
                    session_id: "root".to_string(),
                    state: AgentStateEnvelope::default(),
                    host: Arc::clone(&manager),
                    turn: None,
                    prompt_usage: Some(PromptUsage {
                        prompt_context_tokens: 70,
                        input_tokens: 70,
                        cached_input_tokens: 0,
                        context_budget_tokens: 70,
                    }),
                    max_context_tokens: Some(100),
                    context_folding: Some(ContextFoldingConfig::default()),
                },
                messages,
            )
            .await
            .expect("mutate messages");
        assert!(!compacted.iter().any(|message| {
            message
                .parts
                .iter()
                .any(|part| part.content.contains("oldest user"))
        }));

        let contributions = session
            .collect_prompt_contributions(PromptHookContext {
                session_id: "root".to_string(),
                host: manager,
                prompt: crate::PromptContext {
                    tool_names: vec!["search_history".to_string()],
                    ..Default::default()
                },
                state: AgentStateEnvelope::default(),
            })
            .await
            .expect("prompt");
        assert_eq!(contributions.len(), 2);
        assert!(
            contributions
                .iter()
                .any(|contribution| contribution.content.contains("Older turns were archived"))
        );
    }

    #[cfg(feature = "sqlite-store")]
    #[tokio::test]
    async fn history_plugin_hides_search_history_until_context_is_archived() {
        let store = Arc::new(Store::memory().expect("store"));
        let host = PluginHost::new(vec![Arc::new(BuiltinHistoryPluginFactory::new(store))]);
        let session = host.build_standard_session("root", None).expect("session");
        let manager: Arc<dyn SessionManager> = Arc::new(MockSessionManager);
        let defs = session
            .tools()
            .definitions()
            .into_iter()
            .filter(|def| def.name == "search_history")
            .collect::<Vec<_>>();
        assert_eq!(defs.len(), 1);
        assert!(!defs[0].enabled);
        assert!(!defs[0].injected);

        let surface = session
            .resolve_tool_surface(ToolSurfaceContext {
                session_id: "root".to_string(),
                mode: ExecutionMode::Standard,
                tools: defs.clone(),
            })
            .expect("tool surface");
        assert_eq!(surface.tools.len(), 1);
        assert!(!surface.tools[0].injected);
        assert!(!surface.tools[0].enabled);

        let archived_result = session
            .mutate_messages(
                MessageMutatorContext {
                    hook: MessageMutatorHook::AfterTokenCount,
                    session_id: "root".to_string(),
                    state: AgentStateEnvelope::default(),
                    host: Arc::clone(&manager),
                    turn: None,
                    prompt_usage: Some(PromptUsage {
                        prompt_context_tokens: 70,
                        input_tokens: 70,
                        cached_input_tokens: 0,
                        context_budget_tokens: 70,
                    }),
                    max_context_tokens: Some(100),
                    context_folding: Some(ContextFoldingConfig::default()),
                },
                vec![
                    crate::Message {
                        id: "u1".to_string(),
                        role: MessageRole::User,
                        parts: vec![crate::Part {
                            id: "u1.p0".to_string(),
                            kind: crate::PartKind::Text,
                            content: "oldest user".repeat(20),
                            tool_call_id: None,
                            tool_name: None,
                            prune_state: crate::PruneState::Intact,
                        }],
                        origin: None,
                    },
                    crate::Message {
                        id: "a1".to_string(),
                        role: MessageRole::Assistant,
                        parts: vec![crate::Part {
                            id: "a1.p0".to_string(),
                            kind: crate::PartKind::Text,
                            content: "oldest assistant".repeat(20),
                            tool_call_id: None,
                            tool_name: None,
                            prune_state: crate::PruneState::Intact,
                        }],
                        origin: None,
                    },
                    crate::Message {
                        id: "u2".to_string(),
                        role: MessageRole::User,
                        parts: vec![crate::Part {
                            id: "u2.p0".to_string(),
                            kind: crate::PartKind::Text,
                            content: "older user".repeat(20),
                            tool_call_id: None,
                            tool_name: None,
                            prune_state: crate::PruneState::Intact,
                        }],
                        origin: None,
                    },
                    crate::Message {
                        id: "a2".to_string(),
                        role: MessageRole::Assistant,
                        parts: vec![crate::Part {
                            id: "a2.p0".to_string(),
                            kind: crate::PartKind::Text,
                            content: "older assistant".repeat(20),
                            tool_call_id: None,
                            tool_name: None,
                            prune_state: crate::PruneState::Intact,
                        }],
                        origin: None,
                    },
                    crate::Message {
                        id: "u3".to_string(),
                        role: MessageRole::User,
                        parts: vec![crate::Part {
                            id: "u3.p0".to_string(),
                            kind: crate::PartKind::Text,
                            content: "recent user".repeat(20),
                            tool_call_id: None,
                            tool_name: None,
                            prune_state: crate::PruneState::Intact,
                        }],
                        origin: None,
                    },
                    crate::Message {
                        id: "a3".to_string(),
                        role: MessageRole::Assistant,
                        parts: vec![crate::Part {
                            id: "a3.p0".to_string(),
                            kind: crate::PartKind::Text,
                            content: "recent assistant".repeat(20),
                            tool_call_id: None,
                            tool_name: None,
                            prune_state: crate::PruneState::Intact,
                        }],
                        origin: None,
                    },
                    crate::Message {
                        id: "u4".to_string(),
                        role: MessageRole::User,
                        parts: vec![crate::Part {
                            id: "u4.p0".to_string(),
                            kind: crate::PartKind::Text,
                            content: "latest user".repeat(20),
                            tool_call_id: None,
                            tool_name: None,
                            prune_state: crate::PruneState::Intact,
                        }],
                        origin: None,
                    },
                    crate::Message {
                        id: "a4".to_string(),
                        role: MessageRole::Assistant,
                        parts: vec![crate::Part {
                            id: "a4.p0".to_string(),
                            kind: crate::PartKind::Text,
                            content: "latest assistant".repeat(20),
                            tool_call_id: None,
                            tool_name: None,
                            prune_state: crate::PruneState::Intact,
                        }],
                        origin: None,
                    },
                ],
            )
            .await
            .expect("mutate messages");
        assert!(!archived_result.is_empty());

        let surface = session
            .resolve_tool_surface(ToolSurfaceContext {
                session_id: "root".to_string(),
                mode: ExecutionMode::Standard,
                tools: defs,
            })
            .expect("tool surface");
        assert_eq!(surface.tools.len(), 1);
        assert!(surface.tools[0].injected);
        assert!(surface.tools[0].enabled);
    }

    #[cfg(feature = "sqlite-store")]
    #[test]
    fn search_history_lists_all_without_query_in_stable_turn_order() {
        let store = Arc::new(Store::memory().expect("store"));
        store.history_add_turn(
            "root",
            &json!({
                "index": 2,
                "user_message": "second",
                "prose": "",
                "code": "",
                "output": "",
                "tool_calls": [],
                "files_read": [],
                "files_written": [],
            }),
        );
        store.history_add_turn(
            "root",
            &json!({
                "index": 1,
                "user_message": "first",
                "prose": "",
                "code": "",
                "output": "",
                "tool_calls": [],
                "files_read": [],
                "files_written": [],
            }),
        );
        let tools = HistoryTools::new(store);

        let result = tools.search_history(&json!({ "__agent_id__": "root" }));

        assert!(result.success);
        let items = result.result.as_array().cloned().unwrap_or_default();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["turn"], 1);
        assert_eq!(items[1]["turn"], 2);
        assert!(items[0].get("score").is_none());
        assert!(items[0].get("field_hits").is_none());
    }

    #[tokio::test]
    async fn prompt_context_plugin_contributes_environment_and_project_instruction_sections() {
        let host = PluginHost::new(vec![Arc::new(BuiltinPromptContextPluginFactory::new(
            Arc::new(StaticInstructionSource {
                text: "Repo rules".to_string(),
                read_text: String::new(),
            }),
            PromptContextPluginConfig::default(),
        ))]);
        let session = host.build_standard_session("root", None).expect("session");
        let contributions = session
            .collect_prompt_contributions(PromptHookContext {
                session_id: "root".to_string(),
                host: Arc::new(MockSessionManager),
                prompt: crate::PromptContext::default(),
                state: AgentStateEnvelope::default(),
            })
            .await
            .expect("prompt contributions");

        assert!(contributions.iter().any(|contribution| {
            contribution.section == PromptSectionName::Environment
                && contribution.content.contains("Working directory:")
        }));
        assert!(contributions.iter().any(|contribution| {
            contribution.section == PromptSectionName::Guidance
                && contribution.content == "### Project Instructions\nRepo rules"
        }));
    }

    #[test]
    fn tool_surface_plugin_shapes_search_surface_and_omitted_tool_note() {
        let host = PluginHost::new(vec![Arc::new(StateToolsPluginFactory::new(
            ExecutionMode::Standard,
        ))]);
        let session = host.build_standard_session("root", None).expect("session");

        let surface = session
            .resolve_tool_surface(ToolSurfaceContext {
                session_id: "root".to_string(),
                mode: ExecutionMode::Standard,
                tools: vec![
                    ToolDefinition {
                        name: "search_tools".to_string(),
                        description: "Discover tools".to_string(),
                        params: vec![],
                        returns: "list".to_string(),
                        examples: vec![],
                        enabled: true,
                        injected: false,
                    },
                    ToolDefinition {
                        name: "read_file".to_string(),
                        description: "Read files".to_string(),
                        params: vec![],
                        returns: "str".to_string(),
                        examples: vec![],
                        enabled: true,
                        injected: true,
                    },
                    ToolDefinition {
                        name: "apply_patch".to_string(),
                        description: "Apply patches".to_string(),
                        params: vec![],
                        returns: "str".to_string(),
                        examples: vec![],
                        enabled: true,
                        injected: false,
                    },
                ],
            })
            .expect("tool surface");

        assert!(
            surface
                .tools
                .iter()
                .any(|tool| tool.name == "search_tools" && tool.enabled && !tool.injected)
        );
        assert!(surface.tool_list_notes.iter().any(|note| {
            note.contains("additional tool(s) are available but omitted from this prompt")
        }));
    }

    #[test]
    fn tool_surface_plugin_hides_search_tools_when_nothing_is_omitted() {
        let host = PluginHost::new(vec![Arc::new(StateToolsPluginFactory::new(
            ExecutionMode::Standard,
        ))]);
        let session = host.build_standard_session("root", None).expect("session");

        let surface = session
            .resolve_tool_surface(ToolSurfaceContext {
                session_id: "root".to_string(),
                mode: ExecutionMode::Standard,
                tools: vec![
                    ToolDefinition {
                        name: "search_tools".to_string(),
                        description: "Discover tools".to_string(),
                        params: vec![],
                        returns: "list".to_string(),
                        examples: vec![],
                        enabled: true,
                        injected: false,
                    },
                    ToolDefinition {
                        name: "read_file".to_string(),
                        description: "Read files".to_string(),
                        params: vec![],
                        returns: "str".to_string(),
                        examples: vec![],
                        enabled: true,
                        injected: true,
                    },
                ],
            })
            .expect("tool surface");

        assert_eq!(surface.tools.len(), 2);
        assert!(
            surface
                .tools
                .iter()
                .any(|tool| tool.name == "search_tools" && !tool.enabled && !tool.injected)
        );
        assert!(surface.tool_list_notes.is_empty());
    }

    #[tokio::test]
    async fn plan_tracker_plugin_registers_update_plan_and_restores_state() {
        let host = PluginHost::new(vec![Arc::new(PlanTrackerPluginFactory)]);
        let session = host.build_standard_session("root", None).expect("session");
        let tracker = session.tools();

        let result = tracker
            .execute(
                "update_plan",
                &json!({
                    "explanation": "Mapped the runtime/plugin seam.",
                    "plan": [
                        {"step":"Inspect planning hooks","status":"completed"},
                        {"step":"Split plugin ownership","status":"in_progress"}
                    ]
                }),
            )
            .await;
        assert!(result.success);
        assert_eq!(result.result.as_str(), Some("Plan updated"));

        let snapshot = session.snapshot().expect("snapshot");
        let restored = host
            .build_standard_session("restored", Some(&snapshot))
            .expect("restored");
        let restored_tracker = restored.tools();
        let second_result = restored_tracker
            .execute(
                "update_plan",
                &json!({
                    "plan": [
                        {"step":"Inspect planning hooks","status":"completed"},
                        {"step":"Split plugin ownership","status":"completed"}
                    ]
                }),
            )
            .await;
        assert!(second_result.success);
    }

    #[tokio::test]
    async fn plan_mode_plugin_toggle_and_status_round_trip() {
        let host = PluginHost::new(vec![Arc::new(PlanModePluginFactory::default())]);
        let session = host.build_standard_session("root", None).expect("session");
        let manager: Arc<dyn SessionManager> = Arc::new(MockSessionManager);

        let status = session
            .invoke_external(
                "plan_mode.status",
                json!({}),
                None,
                true,
                Arc::clone(&manager),
            )
            .await
            .expect("status");
        assert!(status.success);
        assert_eq!(
            status.result.get("enabled").and_then(|v| v.as_bool()),
            Some(false)
        );

        let enabled = session
            .invoke_external(
                "plan_mode.enable",
                json!({}),
                None,
                true,
                Arc::clone(&manager),
            )
            .await
            .expect("enable");
        assert!(enabled.success);
        assert_eq!(
            enabled.result.get("enabled").and_then(|v| v.as_bool()),
            Some(true)
        );

        let snapshot = session.snapshot().expect("snapshot");
        let restored = host
            .build_standard_session("restored", Some(&snapshot))
            .expect("restored");
        let restored_status = restored
            .invoke_external("plan_mode.status", json!({}), None, true, manager)
            .await
            .expect("status");
        assert_eq!(
            restored_status
                .result
                .get("enabled")
                .and_then(|v| v.as_bool()),
            Some(true)
        );

        restored
            .restore(&crate::PluginSessionSnapshot::default())
            .expect("reset restore");
        let reset_status = restored
            .invoke_external(
                "plan_mode.status",
                json!({}),
                None,
                true,
                Arc::new(MockSessionManager),
            )
            .await
            .expect("status");
        assert_eq!(
            reset_status.result.get("enabled").and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    #[tokio::test]
    async fn plan_mode_plugin_injects_guidance_and_blocks_implementation_tools() {
        let host = PluginHost::new(vec![Arc::new(PlanModePluginFactory::default())]);
        let session = host.build_standard_session("root", None).expect("session");
        let manager: Arc<dyn SessionManager> = Arc::new(MockSessionManager);

        session
            .invoke_external(
                "plan_mode.enable",
                json!({}),
                None,
                true,
                Arc::clone(&manager),
            )
            .await
            .expect("enable");

        let before_turn = session
            .before_turn(TurnHookContext {
                session_id: "root".to_string(),
                state: AgentStateEnvelope::default(),
                host: Arc::clone(&manager),
            })
            .await
            .expect("before_turn");
        assert!(before_turn.iter().any(|emitted| matches!(
            &emitted.value,
            PluginDirective::EnqueueMessages { messages }
                if messages.iter().any(|message|
                    message.role == MessageRole::System
                        && message.content.contains("Plan Mode (Conversational)")
                        && message.content.contains("PHASE 1 - Ground in the environment"))
        )));
        assert!(before_turn.iter().any(|emitted| {
            emitted.plugin_id == "plan_mode"
                && matches!(
                    &emitted.value,
                    PluginDirective::EmitEvents { events }
                        if events.iter().any(|event| matches!(
                            event,
                            crate::plugin::PluginSurfaceEvent::ModeIndicatorUpsert { label, .. }
                                if label == "plan"
                        ))
                )
        }));

        let allowed_exec = session
            .before_tool_call(ToolCallHookContext {
                session_id: "root".to_string(),
                tool_name: "exec_command".to_string(),
                args: json!({"cmd":"cargo test -q"}),
                host: Arc::clone(&manager),
            })
            .await
            .expect("before_tool_call");
        assert!(!allowed_exec.is_empty());

        let allowed = session
            .before_tool_call(ToolCallHookContext {
                session_id: "root".to_string(),
                tool_name: "read_file".to_string(),
                args: json!({"path":"src/main.rs"}),
                host: Arc::clone(&manager),
            })
            .await
            .expect("before_tool_call");
        assert!(allowed.is_empty());

        let surface = session
            .resolve_tool_surface(ToolSurfaceContext {
                session_id: "root".to_string(),
                mode: ExecutionMode::Standard,
                tools: vec![
                    ToolDefinition {
                        name: "search_tools".to_string(),
                        description: "Discover tools".to_string(),
                        params: vec![],
                        returns: "list".to_string(),
                        examples: vec![],
                        enabled: true,
                        injected: false,
                    },
                    ToolDefinition {
                        name: "update_plan".to_string(),
                        description: "Update plan".to_string(),
                        params: vec![],
                        returns: "str".to_string(),
                        examples: vec![],
                        enabled: true,
                        injected: true,
                    },
                    ToolDefinition {
                        name: "read_file".to_string(),
                        description: "Read files".to_string(),
                        params: vec![],
                        returns: "str".to_string(),
                        examples: vec![],
                        enabled: true,
                        injected: true,
                    },
                ],
            })
            .expect("tool surface");
        assert!(
            surface
                .tools
                .iter()
                .find(|tool| tool.name == "update_plan")
                .is_some_and(|tool| !tool.enabled && !tool.injected)
        );

        let checklist_blocked = session
            .before_tool_call(ToolCallHookContext {
                session_id: "root".to_string(),
                tool_name: "update_plan".to_string(),
                args: json!({
                    "plan": [{"step":"Inspect planning hooks","status":"in_progress"}]
                }),
                host: Arc::clone(&manager),
            })
            .await
            .expect("before_tool_call");
        assert!(checklist_blocked.iter().any(|emitted| matches!(
            &emitted.value,
            PluginDirective::AbortTurn { code, .. } if code == "plan_mode_tool_blocked"
        )));

        let read_allowed = session
            .before_tool_call(ToolCallHookContext {
                session_id: "root".to_string(),
                tool_name: "read_file".to_string(),
                args: json!({
                    "path":"src/main.rs"
                }),
                host: Arc::clone(&manager),
            })
            .await
            .expect("before_tool_call");
        assert!(read_allowed.is_empty());

        session
            .invoke_external(
                "plan_mode.disable",
                json!({}),
                None,
                true,
                Arc::clone(&manager),
            )
            .await
            .expect("disable");
        session
            .invoke_external(
                "plan_mode.enable",
                json!({}),
                None,
                true,
                Arc::clone(&manager),
            )
            .await
            .expect("enable");

        let checkpoint = session
            .at_checkpoint(crate::CheckpointHookContext {
                session_id: "root".to_string(),
                checkpoint: crate::CheckpointKind::AfterWork,
                state: AgentStateEnvelope::default(),
                host: Arc::clone(&manager),
            })
            .await
            .expect("checkpoint");
        assert!(checkpoint.iter().any(|emitted| matches!(
            &emitted.value,
            PluginDirective::EnqueueMessages { messages }
                if messages.iter().any(|message|
                    message.content.contains("Plan Mode (Conversational)")
                        && message.content.contains("Finalization rule"))
        )));
        assert!(checkpoint.iter().any(|emitted| matches!(
            &emitted.value,
            PluginDirective::EnqueueMessages { messages }
                if messages.iter().any(|message| message.content.contains("<proposed_plan>"))
        )));

        session
            .after_turn(TurnResultHookContext {
                session_id: "root".to_string(),
                turn: empty_turn("root"),
                host: manager,
            })
            .await
            .expect("after_turn");
    }

    #[tokio::test]
    async fn plan_mode_plugin_uses_configured_blocked_tool_set() {
        let host = PluginHost::new(vec![Arc::new(PlanModePluginFactory::new(
            PlanModePluginConfig::default().with_blocked_tools(["read_file"]),
        ))]);
        let session = host.build_standard_session("root", None).expect("session");
        let manager: Arc<dyn SessionManager> = Arc::new(MockSessionManager);

        session
            .invoke_external(
                "plan_mode.enable",
                json!({}),
                None,
                true,
                Arc::clone(&manager),
            )
            .await
            .expect("enable");

        let blocked = session
            .before_tool_call(ToolCallHookContext {
                session_id: "root".to_string(),
                tool_name: "read_file".to_string(),
                args: json!({"path":"src/main.rs"}),
                host: Arc::clone(&manager),
            })
            .await
            .expect("before_tool_call");
        assert!(blocked.iter().any(|emitted| matches!(
            &emitted.value,
            PluginDirective::AbortTurn { code, .. } if code == "plan_mode_tool_blocked"
        )));

        let allowed = session
            .before_tool_call(ToolCallHookContext {
                session_id: "root".to_string(),
                tool_name: "update_plan".to_string(),
                args: json!({
                    "plan": [{"step":"Inspect planning hooks","status":"in_progress"}]
                }),
                host: Arc::clone(&manager),
            })
            .await
            .expect("before_tool_call");
        assert!(allowed.is_empty());
    }

    #[tokio::test]
    async fn plan_mode_plugin_suppresses_proposed_plan_tags_and_emits_panel_events() {
        let host = PluginHost::new(vec![Arc::new(PlanModePluginFactory::default())]);
        let session = host.build_standard_session("root", None).expect("session");
        let manager: Arc<dyn SessionManager> = Arc::new(MockSessionManager);

        session
            .invoke_external(
                "plan_mode.enable",
                json!({}),
                None,
                true,
                Arc::clone(&manager),
            )
            .await
            .expect("enable");
        session
            .before_turn(TurnHookContext {
                session_id: "root".to_string(),
                state: AgentStateEnvelope::default(),
                host: Arc::clone(&manager),
            })
            .await
            .expect("before_turn");

        let stream = session
            .transform_assistant_stream(
                "root",
                "Start\n<proposed_plan>\n- Step one\n".to_string(),
                Arc::clone(&manager),
            )
            .await
            .expect("stream");
        assert_eq!(stream.len(), 1);
        assert_eq!(stream[0].plugin_id, "plan_mode");
        assert_eq!(stream[0].value.chunk, "Start\n");
        assert!(stream[0].value.events.iter().any(|event| matches!(
            event,
            crate::plugin::PluginSurfaceEvent::PanelUpsert { title, content, .. }
                if title == "PROPOSED PLAN" && content.contains("Step one")
        )));

        let response = session
            .transform_assistant_response(
                "root",
                crate::llm::types::LlmResponse {
                    full_text: "Start\n<proposed_plan>\n- Step one\n</proposed_plan>\nDone.".into(),
                    deltas: Vec::new(),
                    parts: vec![crate::llm::types::LlmOutputPart::Text {
                        text: "Start\n<proposed_plan>\n- Step one\n</proposed_plan>\nDone.".into(),
                    }],
                    usage: crate::llm::types::LlmUsage::default(),
                    request_body: None,
                    http_summary: None,
                },
                manager,
            )
            .await
            .expect("response");
        assert_eq!(response.len(), 1);
        assert_eq!(response[0].value.response.full_text, "Start\n\nDone.");
    }
}
