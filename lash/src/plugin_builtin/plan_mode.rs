use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use serde_json::json;

use crate::plugin::{
    ExternalOpDef, ExternalOpKind, PluginDirective, PluginError, PluginFactory, PluginRegistrar,
    PluginSessionContext, PluginSnapshotMeta, SessionParam, SessionPlugin, SnapshotReader,
    SnapshotWriter, ToolSurfaceContribution, ToolSurfaceOverride,
};
use crate::{PluginMessage, ToolResult};

fn plan_mode_guidance_message() -> PluginMessage {
    PluginMessage::text(
        crate::MessageRole::System,
        r#"
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
When the answer can be captured as a short list of concrete choices, pass structured `options` instead of embedding pseudo-multiple-choice text in the question body. Reserve free-form asks for cases where the user genuinely needs to type an unconstrained answer.
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
    )
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
