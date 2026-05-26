use super::*;

impl App {
    pub fn queue_pending_steer(&mut self, turn: PreparedTurn) {
        if turn.is_empty() {
            return;
        }
        self.pending_steers.push_back(turn);
    }

    pub fn queue_turn(&mut self, turn: PreparedTurn) {
        if turn.is_empty() {
            return;
        }
        self.queued_turns.push_back(turn);
    }

    pub fn queue_process_wake(&mut self, input: String) {
        if input.trim().is_empty() {
            return;
        }
        self.pending_process_wakes.push_back(input);
    }

    pub fn has_pending_process_wakes(&self) -> bool {
        !self.pending_process_wakes.is_empty()
    }

    pub fn take_pending_process_wakes(&mut self) -> Vec<String> {
        self.pending_process_wakes.drain(..).collect::<Vec<_>>()
    }

    pub fn mark_process_wakes_in_flight(&mut self, wakes: &[String]) {
        self.in_flight_process_wakes.extend(wakes.iter().cloned());
    }

    pub fn acknowledge_process_wakes(&mut self, messages: &[PluginMessage]) {
        for message in messages {
            if !matches!(message.role, MessageRole::System) {
                continue;
            }
            if let Some(idx) = self
                .in_flight_process_wakes
                .iter()
                .position(|candidate| candidate == &message.content)
            {
                let _ = self.in_flight_process_wakes.remove(idx);
            }
        }
    }

    pub fn recycle_unaccepted_process_wakes(&mut self) {
        while let Some(wake) = self.in_flight_process_wakes.pop_back() {
            self.pending_process_wakes.push_front(wake);
        }
    }

    pub fn requeue_front(&mut self, turn: PreparedTurn, pending: bool) {
        if turn.is_empty() {
            return;
        }
        if pending {
            self.pending_steers.push_front(turn);
        } else {
            self.queued_turns.push_front(turn);
        }
    }

    pub fn take_next_queued_turn(&mut self) -> Option<(PreparedTurn, bool)> {
        self.queued_turns.pop_front().map(|turn| (turn, false))
    }

    pub fn take_last_queued_turn(&mut self) -> Option<(PreparedTurn, bool)> {
        self.queued_turns.pop_back().map(|turn| (turn, false))
    }

    pub fn has_queued_messages(&self) -> bool {
        !self.pending_steers.is_empty() || !self.queued_turns.is_empty()
    }

    pub(super) fn take_matching_pending_steer(&mut self, content: &str) -> Option<PreparedTurn> {
        let idx = self.pending_steers.iter().position(|turn| {
            turn.display_text == content
                || turn.effective_text == content
                || (!turn.input_metadata.transforms.is_empty()
                    && content.starts_with(&turn.display_text))
        })?;
        self.pending_steers.remove(idx)
    }
}
