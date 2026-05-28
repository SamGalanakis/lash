use super::*;

impl App {
    pub fn queue_pending_steer(&mut self, turn: PreparedTurn) {
        if turn.is_empty() {
            return;
        }
        self.queues.pending_steers.push_back(turn);
    }

    pub fn queue_turn(&mut self, turn: PreparedTurn) {
        if turn.is_empty() {
            return;
        }
        self.queues.queued_turns.push_back(turn);
    }

    #[cfg(test)]
    pub fn take_next_queued_turn(&mut self) -> Option<(PreparedTurn, bool)> {
        self.queues
            .queued_turns
            .pop_front()
            .map(|turn| (turn, false))
    }

    pub fn take_queued_turn_by_draft_id(&mut self, draft_id: &str) -> Option<PreparedTurn> {
        let idx = self
            .queues
            .queued_turns
            .iter()
            .position(|turn| turn.draft_id == draft_id)?;
        self.queues.queued_turns.remove(idx)
    }

    pub fn take_matching_queued_turn(&mut self, content: &str) -> Option<PreparedTurn> {
        let idx = self
            .queues
            .queued_turns
            .iter()
            .position(|turn| turn.matches_content(content))?;
        self.queues.queued_turns.remove(idx)
    }

    pub fn take_last_queued_turn(&mut self) -> Option<(PreparedTurn, bool)> {
        self.queues
            .queued_turns
            .pop_back()
            .map(|turn| (turn, false))
    }

    pub fn has_queued_messages(&self) -> bool {
        !self.queues.pending_steers.is_empty() || !self.queues.queued_turns.is_empty()
    }

    pub(super) fn take_matching_pending_steer(&mut self, content: &str) -> Option<PreparedTurn> {
        let idx = self
            .queues
            .pending_steers
            .iter()
            .position(|turn| turn.matches_content(content))?;
        self.queues.pending_steers.remove(idx)
    }
}
