use super::*;
use crate::assistant_text::push_assistant_text_block;
#[cfg(test)]
use lash_core::SessionEvent;
use lash_core::{PluginRuntimeEvent, TurnActivity, TurnEvent};

fn runtime_status_from_plugin_event(
    event: &PluginRuntimeEvent,
) -> Option<(String, Option<String>, std::time::Duration)> {
    match event {
        PluginRuntimeEvent::Status { label, detail, .. } => Some((
            label.clone(),
            detail.clone(),
            std::time::Duration::from_millis(8_000),
        )),
        _ => None,
    }
}

impl App {
    fn activity_renders_prompt_response_inline(activity: &ActivityBlock) -> bool {
        matches!(
            activity.result.artifact,
            Some(ActivityArtifact::QuestionPanel(_))
        )
    }

    pub(super) fn push_prompt_response_user_block(&mut self, display: String) {
        if display.trim().is_empty() {
            return;
        }
        let changed_idx = self.timeline.len();
        let invalidate_from = self.append_invalidation_start();
        self.timeline.push(UiTimelineItem::UserInput(display));
        self.invalidate_height_cache_from(
            invalidate_from
                .min(changed_idx)
                .min(self.timeline.len().saturating_sub(1)),
        );
        self.keep_latest_user_block_visible();
    }

    fn push_activity_block(&mut self, activity: ActivityBlock) {
        let invalidate_from = self.append_invalidation_start();
        let prior_len = self.timeline.len();
        ActivityState::append_projected_activity_to_timeline(&mut self.timeline, activity);
        if !self.timeline.is_empty() {
            let changed_idx = if self.timeline.len() == prior_len {
                prior_len.saturating_sub(1)
            } else {
                prior_len
            };
            self.invalidate_height_cache_from(
                invalidate_from
                    .min(changed_idx)
                    .min(self.timeline.len().saturating_sub(1)),
            );
        }
    }

    fn accept_injected_turn_input(&mut self, messages: &[PluginMessage]) {
        self.finalize_live_markdown();
        let mut accepted_user_message = false;
        for message in messages {
            if !matches!(message.role, MessageRole::User) {
                continue;
            }
            accepted_user_message = true;
            if let Some(turn) = self.take_matching_pending_steer(&message.content) {
                self.push_prepared_user_input(&turn);
            } else {
                self.push_user_input_history_text(message.content.clone());
            }
        }
        if accepted_user_message {
            self.keep_latest_user_block_visible();
        }
    }

    fn commit_injected_messages(&mut self, messages: &[PluginMessage]) {
        self.finalize_live_markdown();
        let mut committed_user_message = false;
        for message in messages {
            match message.role {
                MessageRole::User => {
                    committed_user_message = true;
                    if let Some(turn) = self.take_matching_pending_steer(&message.content) {
                        self.push_prepared_user_input(&turn);
                    } else {
                        self.push_user_input_history_text(message.content.clone());
                    }
                }
                MessageRole::System => {
                    self.timeline
                        .push(UiTimelineItem::SystemMessage(message.content.clone()));
                }
                _ => continue,
            }
        }
        if !messages.is_empty() {
            self.invalidate_height_cache();
            if committed_user_message {
                self.keep_latest_user_block_visible();
            } else {
                self.scroll_to_bottom();
            }
        }
    }

    pub fn push_prepared_user_input(&mut self, turn: &PreparedTurn) {
        self.push_user_input_history_text(turn.history_text());
    }

    pub(super) fn push_user_input_history_text(&mut self, history_text: String) {
        let history_text = history_text.trim().to_string();
        if history_text.is_empty() {
            return;
        }
        let scan_start = self
            .timeline
            .iter()
            .rposition(|block| {
                matches!(
                    block,
                    UiTimelineItem::TurnStart(turn) if turn.role == TurnRole::User
                )
            })
            .unwrap_or(0);
        if self.timeline[scan_start..].iter().any(
            |block| matches!(block, UiTimelineItem::UserInput(text) if text.trim() == history_text),
        ) {
            return;
        }
        let changed_idx = self.timeline.len();
        let invalidate_from = self.append_invalidation_start();
        self.timeline.push_user_turn_start();
        self.timeline.push(UiTimelineItem::UserInput(history_text));
        self.invalidate_height_cache_from(
            invalidate_from
                .min(changed_idx)
                .min(self.timeline.len().saturating_sub(1)),
        );
    }

    /// Process a semantic turn activity, updating display blocks.
    pub fn handle_turn_activity(&mut self, activity: TurnActivity) {
        match activity.event {
            TurnEvent::ReasoningDelta { text } => {
                if text.is_empty() {
                    return;
                }
                self.mark_first_token_arrived();
                if self.live_assistant.has_renderable_output()
                    && self.merge_into_trailing_reasoning_block(&text)
                {
                    self.scroll_to_bottom();
                    return;
                }
                let had_output = self.live_reasoning.has_renderable_output();
                self.live_reasoning.append(&text);
                if !had_output && self.live_reasoning.has_renderable_output() {
                    self.mark_visible_output();
                }
                self.mark_visible_output();
                self.scroll_to_bottom();
            }
            TurnEvent::AssistantProseDelta { text } => {
                self.mark_first_token_arrived();
                self.live_output_chars_estimate += text.chars().count() as i64;
                self.live_output_tokens_estimate =
                    live::estimate_tokens_from_char_count(self.live_output_chars_estimate);
                if self.live_reasoning.has_renderable_output() {
                    self.commit_live_reasoning_block();
                }
                let had_output = self.live_assistant.has_renderable_output();
                self.live_assistant.append(&text);
                if !had_output && self.live_assistant.has_renderable_output() {
                    self.mark_visible_output();
                }
                self.scroll_to_bottom();
            }
            TurnEvent::SubmittedValue { value } => {
                self.finalize_live_markdown();
                let text = self::projection::render_submitted_value(&value);
                if push_assistant_text_block(&mut self.timeline, &text) {
                    self.mark_first_token_arrived();
                    self.live_output_chars_estimate += text.chars().count() as i64;
                    self.live_output_tokens_estimate =
                        live::estimate_tokens_from_char_count(self.live_output_chars_estimate);
                    self.mark_visible_output();
                    self.invalidate_height_cache();
                    self.scroll_to_bottom();
                }
            }
            TurnEvent::ToolCallCompleted {
                name,
                args,
                output,
                duration_ms,
                ..
            } => {
                self.finalize_live_markdown();
                self.clear_live_tool_output();
                let activities =
                    self.activity_state
                        .project_tool_output(&name, args, output, duration_ms);
                let renders_prompt_response_inline = activities
                    .iter()
                    .any(Self::activity_renders_prompt_response_inline);
                if renders_prompt_response_inline || name == "plan_exit" {
                    self.pending_option_prompt_response = None;
                } else if let Some(display) = self.pending_option_prompt_response.take() {
                    self.push_prompt_response_user_block(display);
                }
                if let Some(activity) = activities.last() {
                    let detail = activity.result.detail_lines.first().cloned();
                    self.set_status(activity.call.summary.clone(), detail, true);
                } else if !is_batch_tool_name(&name) {
                    self.set_status(name.clone(), None, true);
                }
                for activity in activities {
                    self.push_activity_block(activity);
                }
                if !matches!(self.timeline.last(), Some(UiTimelineItem::Splash)) {
                    self.mark_visible_output();
                }
                self.scroll_to_bottom();
            }
            TurnEvent::ToolCallStarted {
                call_id,
                name,
                args,
            } => {
                self.finalize_live_markdown();
                let title = live::live_tool_output_title(&name, &args);
                self.live_tool_output.start(call_id, title);
                self.invalidate_live_tool_output_cache();
            }
            TurnEvent::CodeBlockStarted { code, .. } => {
                self.finalize_live_markdown();
                let changed_idx = self.timeline.len();
                let invalidate_from = self.append_invalidation_start();
                self.timeline.push(UiTimelineItem::LashlangCode(code));
                self.invalidate_height_cache_from(
                    invalidate_from
                        .min(changed_idx)
                        .min(self.timeline.len().saturating_sub(1)),
                );
                self.invalidate_live_tool_output_cache();
                self.scroll_to_bottom();
            }
            TurnEvent::ModelRequestStarted { mode_iteration } => {
                self.finalize_live_markdown();
                self.iteration = mode_iteration + 1;
                if let Some(detail) = self.pending_retry_status.take() {
                    self.set_status("retrying", Some(detail), true);
                } else {
                    self.set_status("thinking", None, true);
                }
                self.live_output_chars_estimate = 0;
                self.live_output_tokens_estimate = 0;
                self.keep_latest_user_block_visible();
            }
            TurnEvent::RetryStatus {
                wait_seconds,
                attempt,
                max_attempts,
                reason,
            } => {
                let mut reason_short: String = reason.chars().take(60).collect();
                if reason.chars().count() > 60 {
                    reason_short.push_str("...");
                }
                let retry_detail =
                    format!("attempt {}/{} · {}", attempt, max_attempts, reason_short);
                self.pending_retry_status = Some(retry_detail.clone());
                self.set_status(
                    "retrying",
                    Some(format!("in {}s · {}", wait_seconds, retry_detail)),
                    true,
                );
                self.scroll_to_bottom();
            }
            TurnEvent::Error { message } => {
                self.finalize_live_markdown();
                if crate::util::is_cancelled_error(&message, None) {
                    let manual_interrupt_requested = self.manual_interrupt_requested;
                    self.stop_turn();
                    self.timeline
                        .push_system_message_if_new(if manual_interrupt_requested {
                            crate::util::manual_interrupt_message().to_string()
                        } else {
                            "Cancelled.".to_string()
                        });
                } else {
                    self.manual_interrupt_requested = false;
                    self.set_transient_status(
                        "error",
                        Some(message.chars().take(96).collect()),
                        std::time::Duration::from_secs(8),
                    );
                    let changed_idx = self.timeline.len();
                    let invalidate_from = self.append_invalidation_start();
                    self.timeline.push(UiTimelineItem::Error(message));
                    self.invalidate_height_cache_from(
                        invalidate_from
                            .min(changed_idx)
                            .min(self.timeline.len().saturating_sub(1)),
                    );
                }
                self.mark_visible_output();
                self.scroll_to_bottom();
            }
            TurnEvent::Usage {
                usage, cumulative, ..
            } => {
                let should_clear_live_estimate = usage.output_tokens > 0
                    || usage.reasoning_tokens > 0
                    || usage.cached_input_tokens > 0;
                self.last_response_usage = usage.clone();
                self.parent_session_cumulative = cumulative;
                self.recompute_session_token_usage();
                if should_clear_live_estimate {
                    self.live_output_chars_estimate = 0;
                    self.live_output_tokens_estimate = 0;
                }
            }
            TurnEvent::ChildUsage {
                session_id,
                cumulative,
                ..
            } => {
                self.child_session_cumulatives
                    .insert(session_id, cumulative);
                self.recompute_session_token_usage();
            }
            TurnEvent::PluginRuntime { plugin_id, event } => {
                if let Some((status, detail, duration)) = runtime_status_from_plugin_event(&event) {
                    self.set_transient_status(status, detail, duration);
                    self.dirty = true;
                }
                let renders_visible_output =
                    crate::plugin_surface::event_renders_visible_output(&event);
                let mutation = crate::plugin_surface::apply_surface_event(
                    &mut self.timeline,
                    &mut self.plugin_mode_indicators,
                    &self.plan_dock,
                    &plugin_id,
                    event,
                );
                if mutation.blocks_changed {
                    self.invalidate_height_cache();
                    if renders_visible_output {
                        self.mark_visible_output();
                    }
                    self.scroll_to_bottom();
                }
                if mutation.indicators_changed {
                    self.dirty = true;
                }
                if let Some(next_dock) = mutation.plan_dock_change {
                    self.plan_dock = next_dock.filter(|state| !state.is_empty());
                    self.invalidate_height_cache();
                    self.dirty = true;
                }
            }
            TurnEvent::QueuedInputAccepted { inputs, .. } => {
                let messages = inputs
                    .iter()
                    .map(|input| input.message.clone())
                    .collect::<Vec<_>>();
                self.accept_injected_turn_input(&messages);
            }
            TurnEvent::QueuedMessagesCommitted { messages, .. } => {
                self.commit_injected_messages(&messages);
            }
            TurnEvent::CodeBlockCompleted { .. } | TurnEvent::ToolValue { .. } => {}
        }
    }

    #[cfg(test)]
    pub fn handle_session_event(&mut self, event: SessionEvent) {
        if matches!(event, SessionEvent::Done) {
            self.finalize_live_markdown();
            self.stop_turn();
            self.scroll_to_bottom();
        } else if let SessionEvent::Message { text, kind } = &event
            && kind == "tool_output"
        {
            let current_status = self
                .live_turn
                .as_ref()
                .map(|turn| turn.status_text.as_str());
            let stream_active =
                self.running || current_status.is_some_and(|status| status.contains("shell"));
            if stream_active {
                self.push_test_tool_output(text);
                self.invalidate_live_tool_output_cache();
                self.mark_visible_output();
                self.scroll_to_bottom();
            }
        } else if let Some(activity) = test_session_event_to_turn_activity(event) {
            self.handle_turn_activity(activity);
        }
    }
}

#[cfg(test)]
fn test_session_event_to_turn_activity(event: SessionEvent) -> Option<TurnActivity> {
    let turn_event = match event {
        SessionEvent::LlmRequest { mode_iteration, .. } => {
            TurnEvent::ModelRequestStarted { mode_iteration }
        }
        SessionEvent::TextDelta { content } => TurnEvent::AssistantProseDelta { text: content },
        SessionEvent::ReasoningDelta { content } => TurnEvent::ReasoningDelta { text: content },
        SessionEvent::ToolCallStart {
            call_id,
            name,
            args,
        } => TurnEvent::ToolCallStarted {
            call_id,
            name,
            args,
        },
        SessionEvent::ToolCall {
            call_id,
            name,
            args,
            output,
            duration_ms,
        } => TurnEvent::ToolCallCompleted {
            call_id,
            name,
            args,
            output,
            duration_ms,
        },
        SessionEvent::Message { text, kind } if kind == "lashlang_code" => {
            TurnEvent::CodeBlockStarted {
                language: "lashlang".to_string(),
                code: text,
            }
        }
        SessionEvent::TokenUsage {
            mode_iteration,
            usage,
            cumulative,
        } => TurnEvent::Usage {
            mode_iteration,
            usage,
            cumulative,
        },
        SessionEvent::RetryStatus {
            wait_seconds,
            attempt,
            max_attempts,
            reason,
            ..
        } => TurnEvent::RetryStatus {
            wait_seconds,
            attempt,
            max_attempts,
            reason,
        },
        SessionEvent::Error { message, .. } => TurnEvent::Error { message },
        SessionEvent::PluginEvent { plugin_id, event } => {
            TurnEvent::PluginRuntime { plugin_id, event }
        }
        SessionEvent::InjectedTurnInputAccepted { inputs, checkpoint } => {
            TurnEvent::QueuedInputAccepted { checkpoint, inputs }
        }
        SessionEvent::InjectedMessagesCommitted {
            messages,
            checkpoint,
        } => TurnEvent::QueuedMessagesCommitted {
            messages,
            checkpoint,
        },
        SessionEvent::Done
        | SessionEvent::ChildTokenUsage { .. }
        | SessionEvent::TurnOutcome { .. }
        | SessionEvent::LlmResponse { .. }
        | SessionEvent::Message { .. } => return None,
    };
    Some(TurnActivity::independent(turn_event))
}
