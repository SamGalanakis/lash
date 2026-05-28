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
