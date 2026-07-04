use super::*;
use crate::assistant_text::{push_assistant_reasoning_block, push_assistant_text_block};

#[cfg(test)]
const STREAMING_OUTPUT_MAX_LINES: usize = 48;
#[cfg(test)]
const STREAMING_OUTPUT_LINE_CHAR_LIMIT: usize = 240;

pub struct LiveTurnState {
    pub run_state: CliRunState,
    pub status_detail: Option<String>,
    pub phase_started_at: std::time::Instant,
    pub turn_started_at: std::time::Instant,
    pub has_visible_user_input: bool,
    pub has_visible_output: bool,
    pub output_start_anchor_pending: bool,
    pub transient_until: Option<std::time::Instant>,
}

impl LiveTurnState {
    pub(super) fn new(run_state: CliRunState, status_detail: Option<String>) -> Self {
        let now = std::time::Instant::now();
        Self {
            run_state,
            status_detail,
            phase_started_at: now,
            turn_started_at: now,
            has_visible_user_input: false,
            has_visible_output: false,
            output_start_anchor_pending: false,
            transient_until: None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LiveToolOutput {
    #[serde(default)]
    pub lines: Vec<String>,
    #[serde(default)]
    pub hidden: usize,
    #[serde(default)]
    pub partial: String,
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
}

impl LiveToolOutput {
    pub(super) fn clear(&mut self) {
        self.lines.clear();
        self.hidden = 0;
        self.partial.clear();
        self.call_id = None;
        self.title = None;
    }

    pub(super) fn start(&mut self, call_id: Option<String>, title: String) {
        self.clear();
        self.call_id = call_id;
        self.title = Some(title);
    }

    pub(crate) fn height(&self) -> usize {
        usize::from(self.hidden > 0) + self.lines.len() + usize::from(!self.partial.is_empty())
    }

    #[cfg(test)]
    pub(super) fn push_text(&mut self, text: &str, line_char_limit: usize, max_lines: usize) {
        let sanitized = strip_ansi_escape_sequences(text);
        let mut chars = sanitized.chars().peekable();
        while let Some(ch) = chars.next() {
            match ch {
                '\r' if matches!(chars.peek(), Some('\n')) => {
                    chars.next();
                    let completed = std::mem::take(&mut self.partial);
                    self.push_line(completed, line_char_limit, max_lines);
                }
                '\r' => {
                    self.partial.clear();
                }
                '\n' => {
                    let completed = std::mem::take(&mut self.partial);
                    self.push_line(completed, line_char_limit, max_lines);
                }
                '\t' => {
                    if !self.partial.chars().last().is_some_and(char::is_whitespace) {
                        self.partial.push(' ');
                    }
                }
                '\u{8}' | '\u{7f}' => {
                    self.partial.pop();
                }
                control if control.is_control() => {}
                _ => self.partial.push(ch),
            }

            if self.partial.chars().count() > line_char_limit {
                self.partial = smart_truncate_preview_line(&self.partial, line_char_limit);
            }
        }
    }

    #[cfg(test)]
    fn push_line(&mut self, line: String, line_char_limit: usize, max_lines: usize) {
        if self.lines.len() == max_lines {
            self.lines.remove(0);
            self.hidden += 1;
        }
        self.lines
            .push(smart_truncate_preview_line(&line, line_char_limit));
    }
}

pub(super) fn live_tool_output_title(name: &str, args: &serde_json::Value) -> String {
    fn arg_str<'a>(args: &'a serde_json::Value, key: &str) -> Option<&'a str> {
        args.get(key).and_then(|value| value.as_str())
    }

    match name {
        "exec_command" | "start_command" => arg_str(args, "cmd")
            .map(str::to_string)
            .unwrap_or_else(|| "shell output".to_string()),
        "write_stdin" => {
            if arg_str(args, "chars").is_some_and(str::is_empty) {
                "poll command output".to_string()
            } else {
                "write to command".to_string()
            }
        }
        other => other.replace('_', " "),
    }
}

pub(super) fn estimate_tokens_from_char_count(chars: i64) -> i64 {
    if chars <= 0 { 0 } else { (chars + 3) / 4 }
}

impl App {
    pub(super) fn ensure_live_turn(&mut self) -> &mut LiveTurnState {
        let run_state = self.run_state;
        self.live
            .turn
            .get_or_insert_with(|| LiveTurnState::new(run_state, None))
    }

    pub fn start_turn(&mut self) {
        self.foreground_turn_active = true;
        self.run_state = CliRunState::Working;
        self.manual_interrupt_requested = false;
        self.pending_retry_status = None;
        self.active_ui_turn_ordinal = Some(self.latest_ui_turn_ordinal());
        self.next_lashlang_block_ordinal = 0;
        self.lashlang_block_anchors.clear();
        self.iteration = 0;
        self.live.assistant.clear();
        self.live.reasoning.clear();
        self.clear_live_tool_output();
        self.usage.live_output_chars_estimate = 0;
        self.usage.live_output_tokens_estimate = 0;
        self.live.turn = Some(LiveTurnState::new(CliRunState::Working, None));
        self.follow_mode = FollowOutputMode::PinnedTurnStart;
    }

    pub(crate) fn mark_live_turn_user_input_visible(&mut self) {
        if let Some(turn) = self.live.turn.as_mut() {
            turn.has_visible_user_input = true;
        }
    }

    pub(crate) fn active_turn_has_visible_user_input(&self) -> bool {
        self.live
            .turn
            .as_ref()
            .is_some_and(|turn| turn.has_visible_user_input)
    }

    pub(crate) fn turn_active(&self) -> bool {
        self.foreground_turn_active
    }

    pub(crate) fn can_inject_into_active_turn(&self) -> bool {
        self.foreground_turn_active
            && self.run_state.is_injectable_runtime_phase()
            && self.active_turn_has_visible_user_input()
    }

    pub(crate) fn route_turn_submission(&self, runtime_available: bool) -> TurnSubmissionRoute {
        if !runtime_available {
            return TurnSubmissionRoute::BlockedSessionSwitch;
        }
        if !self.foreground_turn_active {
            return TurnSubmissionRoute::SendNow;
        }
        if self.can_inject_into_active_turn() {
            TurnSubmissionRoute::InjectActiveTurn
        } else {
            TurnSubmissionRoute::QueueNextFullTurn
        }
    }

    pub fn stop_turn(&mut self) {
        self.invalidate_live_reasoning_tail();
        self.foreground_turn_active = false;
        self.manual_interrupt_requested = false;
        self.pending_retry_status = None;
        self.active_ui_turn_ordinal = None;
        self.next_lashlang_block_ordinal = 0;
        self.lashlang_block_anchors.clear();
        self.live.reasoning.clear();
        self.live.assistant.clear();
        self.clear_live_tool_output();
        self.usage.live_output_chars_estimate = 0;
        self.usage.live_output_tokens_estimate = 0;
        if self.follow_mode == FollowOutputMode::PinnedTurnStart {
            self.follow_mode = FollowOutputMode::PinnedBottom;
        }
        if let Some(display) = self.queues.pending_option_prompt_response.take() {
            self.push_prompt_response_user_block(display);
        }
        let keep_transient = self
            .live
            .turn
            .as_ref()
            .and_then(|turn| turn.transient_until)
            .is_some_and(|until| until > std::time::Instant::now());
        if !keep_transient {
            self.live.turn = None;
            self.run_state = CliRunState::Idle;
        }
        self.dirty = true;
    }

    pub(super) fn set_status(
        &mut self,
        state: CliRunState,
        details: Option<String>,
        reset_timer: bool,
    ) {
        self.run_state = state;
        let turn = self.ensure_live_turn();
        let changed = turn.run_state != state || turn.status_detail != details;
        turn.run_state = state;
        turn.status_detail = details;
        turn.transient_until = None;
        if changed || reset_timer {
            turn.phase_started_at = std::time::Instant::now();
        }
    }

    pub(super) fn set_transient_status(
        &mut self,
        state: CliRunState,
        details: Option<String>,
        duration: std::time::Duration,
    ) {
        self.run_state = state;
        let now = std::time::Instant::now();
        let turn = self.ensure_live_turn();
        turn.run_state = state;
        turn.status_detail = details;
        turn.phase_started_at = now;
        turn.transient_until = Some(now + duration);
    }

    pub(super) fn clear_status(&mut self) {
        self.foreground_turn_active = false;
        self.manual_interrupt_requested = false;
        self.pending_retry_status = None;
        self.live.turn = None;
        self.run_state = CliRunState::Idle;
    }

    pub(crate) fn sync_foreground_turn_active(&mut self, active: bool) {
        if self.foreground_turn_active == active {
            return;
        }
        self.foreground_turn_active = active;
        if !active && self.run_state.is_runtime_active() {
            let keep_transient_error = self.live.turn.as_ref().is_some_and(|turn| {
                turn.run_state == CliRunState::Error
                    && turn
                        .transient_until
                        .is_some_and(|until| until > std::time::Instant::now())
            });
            if !keep_transient_error {
                self.live.turn = None;
                self.run_state = CliRunState::Idle;
            }
        }
        self.dirty = true;
    }

    pub fn note_manual_interrupt_requested(&mut self) {
        self.manual_interrupt_requested = true;
    }

    pub(super) fn mark_first_token_arrived(&mut self) {
        if self
            .live
            .turn
            .as_ref()
            .is_some_and(|turn| turn.run_state == CliRunState::Thinking)
        {
            self.set_status(CliRunState::Responding, None, true);
        }
    }

    pub(super) fn mark_visible_output(&mut self) {
        if let Some(turn) = self.live.turn.as_mut()
            && !turn.has_visible_output
        {
            turn.has_visible_output = true;
            turn.output_start_anchor_pending =
                matches!(self.follow_mode, FollowOutputMode::PinnedTurnStart);
        }
    }

    pub(super) fn ensure_live_markdown_rendered(&mut self, viewport_width: usize) {
        self.live.reasoning.ensure_rendered(viewport_width);
        self.live.assistant.ensure_rendered(viewport_width);
    }

    pub fn live_reasoning_lines_snapshot(&self) -> Option<&[Line<'static>]> {
        (!self.live.reasoning.lines().is_empty()).then_some(self.live.reasoning.lines())
    }

    pub fn live_assistant_lines_snapshot(&self) -> Option<&[Line<'static>]> {
        (!self.live.assistant.lines().is_empty()).then_some(self.live.assistant.lines())
    }

    pub(crate) fn has_live_markdown_output(&self) -> bool {
        self.live.reasoning.has_renderable_output() || self.live.assistant.has_renderable_output()
    }

    pub(crate) fn live_reasoning_leading_padding(&self) -> usize {
        if !self.live.reasoning.has_renderable_output() {
            return 0;
        }

        match self.timeline.last() {
            Some(
                UiTimelineItem::AssistantReasoning(_)
                | UiTimelineItem::Splash
                | UiTimelineItem::TurnStart(_),
            )
            | None => 0,
            _ => 1,
        }
    }

    pub(crate) fn live_assistant_leading_padding(&self) -> usize {
        if !self.live.assistant.has_renderable_output() {
            return 0;
        }
        if self.live.reasoning.has_renderable_output() {
            return 0;
        }

        match self.timeline.last() {
            Some(
                UiTimelineItem::AssistantText(_)
                | UiTimelineItem::AssistantReasoning(_)
                | UiTimelineItem::Splash
                | UiTimelineItem::TurnStart(_),
            )
            | None => 0,
            _ => 1,
        }
    }

    pub(super) fn live_reasoning_height(&self) -> usize {
        let lines = self.live.reasoning.lines();
        if lines.is_empty() {
            return 0;
        }
        self.live_reasoning_leading_padding() + lines.len()
    }

    pub(super) fn live_assistant_height(&self) -> usize {
        let lines = self.live.assistant.lines();
        if lines.is_empty() {
            return 0;
        }
        self.live_assistant_leading_padding() + lines.len()
    }

    pub(crate) fn live_tool_output_anchor_block_index(&self) -> Option<usize> {
        if self.live.tool_output.height() == 0 {
            return None;
        }
        self.timeline
            .iter()
            .enumerate()
            .rev()
            .find_map(|(idx, block)| match block {
                UiTimelineItem::Activity(activity)
                    if matches!(
                        activity.call.kind,
                        ActivityKind::ShellCommand
                            | ActivityKind::ShellInteraction
                            | ActivityKind::Subagent
                    ) && matches!(
                        activity.result.status,
                        ActivityStatus::Running | ActivityStatus::Partial
                    ) =>
                {
                    Some(idx)
                }
                _ => None,
            })
    }

    pub(super) fn invalidate_live_tool_output_cache(&mut self) {
        if let Some(idx) = self.live_tool_output_anchor_block_index() {
            self.invalidate_height_cache_from(idx);
        }
    }

    pub(super) fn clear_live_tool_output(&mut self) {
        let had_output = self.live.tool_output.height() > 0;
        let anchor_idx = self.live_tool_output_anchor_block_index();
        self.live.tool_output.clear();
        if had_output && let Some(idx) = anchor_idx {
            self.invalidate_height_cache_from(idx);
        }
    }

    pub(super) fn reconcile_trailing_assistant_block(&mut self, text: &str) -> bool {
        let Some(UiTimelineItem::AssistantText(existing)) = self.timeline.last_mut() else {
            return false;
        };
        if text.starts_with(existing.as_str()) {
            if *existing != text {
                *existing = text.to_string();
                let idx = self.timeline.len().saturating_sub(1);
                self.invalidate_height_cache_from(idx);
            }
            return true;
        }
        if existing.is_empty() || existing.starts_with(text) {
            return true;
        }
        false
    }

    pub(super) fn commit_live_assistant_block(&mut self) {
        let Some(cleaned) = self.live.assistant.take_normalized_text() else {
            return;
        };

        if self.reconcile_trailing_assistant_block(&cleaned) {
            self.mark_visible_output();
            return;
        }

        let changed_idx = self.timeline.len();
        let invalidate_from = self.append_invalidation_start();
        if push_assistant_text_block(&mut self.timeline, &cleaned) {
            self.invalidate_height_cache_from(
                invalidate_from
                    .min(changed_idx)
                    .min(self.timeline.len().saturating_sub(1)),
            );
            self.mark_visible_output();
        }
    }

    pub(super) fn commit_live_reasoning_block(&mut self) {
        let Some(cleaned) = self.live.reasoning.take_normalized_text() else {
            return;
        };
        let prior_len = self.timeline.len();
        if push_assistant_reasoning_block(&mut self.timeline, &cleaned) {
            let changed_idx = if self.timeline.len() == prior_len {
                prior_len.saturating_sub(1)
            } else {
                prior_len
            };
            self.invalidate_height_cache_from(changed_idx.min(self.timeline.len() - 1));
            self.mark_visible_output();
        }
    }

    pub(super) fn finalize_live_markdown(&mut self) {
        self.commit_live_reasoning_block();
        self.commit_live_assistant_block();
    }

    #[cfg(test)]
    pub(super) fn push_test_tool_output(&mut self, text: &str) {
        self.live.tool_output.push_text(
            text,
            STREAMING_OUTPUT_LINE_CHAR_LIMIT,
            STREAMING_OUTPUT_MAX_LINES,
        );
    }
}
