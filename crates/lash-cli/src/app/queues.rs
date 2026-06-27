use super::*;

fn source_key_draft_id(source_key: &str) -> &str {
    source_key
        .strip_prefix("host:")
        .or_else(|| source_key.strip_prefix("injection:"))
        .unwrap_or(source_key)
}

fn pending_turn_input_signature(
    inputs: &[lash_core::PendingTurnInput],
) -> Vec<(String, Option<String>, lash_core::TurnInputState)> {
    inputs
        .iter()
        .map(|input| {
            (
                input.input_id.clone(),
                input.source_key.clone(),
                input.state,
            )
        })
        .collect()
}

fn pending_turn_input_ids(
    inputs: &[lash_core::PendingTurnInput],
) -> std::collections::HashSet<String> {
    inputs.iter().map(|input| input.input_id.clone()).collect()
}

pub(crate) fn turn_input_display_text(input: &lash_core::TurnInput) -> String {
    input
        .items
        .iter()
        .filter_map(|item| match item {
            lash_core::InputItem::Text { text } => Some(text.as_str()),
            lash_core::InputItem::ImageRef { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

pub(crate) fn prepared_turn_from_queued_input(
    input: &lash_core::TurnInput,
) -> Option<PreparedTurn> {
    let text = turn_input_display_text(input);
    let images = input
        .items
        .iter()
        .filter_map(|item| match item {
            lash_core::InputItem::ImageRef { id } => input.image_blobs.get(id),
            lash_core::InputItem::Text { .. } => None,
        })
        .enumerate()
        .map(|(idx, png_bytes)| PendingImage {
            id: idx + 1,
            png_bytes: png_bytes.clone(),
        })
        .collect::<Vec<_>>();
    if text.trim().is_empty() && images.is_empty() {
        return None;
    }
    Some(PreparedTurn::prepare_with_effective_text(
        text.clone(),
        text,
        images,
    ))
}

impl Queues {
    fn take_by_draft_id(&mut self, draft_id: &str) -> Option<PreparedTurn> {
        self.draft_presentations.remove(draft_id)
    }

    fn take_matching_content(&mut self, content: &str) -> Option<PreparedTurn> {
        let id = self
            .draft_presentations
            .iter()
            .find_map(|(id, turn)| turn.matches_content(content).then(|| id.clone()))?;
        self.draft_presentations.remove(&id)
    }

    fn presentation_for_input(
        &self,
        draft_id: Option<&str>,
        input: &lash_core::TurnInput,
    ) -> Option<PreparedTurn> {
        if let Some(draft_id) = draft_id
            && let Some(turn) = self.draft_presentations.get(draft_id)
        {
            return Some(turn.clone());
        }
        let content = turn_input_display_text(input);
        self.draft_presentations
            .values()
            .find(|turn| turn.matches_content(&content))
            .cloned()
            .or_else(|| prepared_turn_from_queued_input(input))
    }
}

impl App {
    pub fn cache_draft_presentation(&mut self, turn: PreparedTurn) {
        if turn.is_empty() {
            return;
        }
        self.queues
            .draft_presentations
            .insert(turn.draft_id.clone(), turn);
    }

    pub(super) fn take_draft_presentation_by_id(&mut self, draft_id: &str) -> Option<PreparedTurn> {
        self.queues.take_by_draft_id(draft_id)
    }

    pub(super) fn take_matching_draft_presentation(
        &mut self,
        content: &str,
    ) -> Option<PreparedTurn> {
        self.queues.take_matching_content(content)
    }

    pub(super) fn take_draft_presentation_for_input(
        &mut self,
        id: Option<&str>,
        content: &str,
    ) -> Option<PreparedTurn> {
        if let Some(id) = id
            && let Some(turn) = self.take_draft_presentation_by_id(id)
        {
            return Some(turn);
        }
        self.take_matching_draft_presentation(content)
    }

    pub fn set_pending_turn_input_snapshot(&mut self, inputs: Vec<lash_core::PendingTurnInput>) {
        let visible_input_ids = pending_turn_input_ids(&inputs);
        let suppressed_before = self.queues.suppressed_preview_input_ids.len();
        self.queues
            .suppressed_preview_input_ids
            .retain(|input_id| visible_input_ids.contains(input_id));
        let suppressed_changed =
            self.queues.suppressed_preview_input_ids.len() != suppressed_before;
        if pending_turn_input_signature(&self.queues.pending_turn_input_snapshot)
            == pending_turn_input_signature(&inputs)
        {
            if suppressed_changed {
                self.dirty = true;
            }
            return;
        }
        self.queues.pending_turn_input_snapshot = inputs;
        self.dirty = true;
    }

    pub fn clear_pending_turn_input_snapshot(&mut self) {
        if self.queues.pending_turn_input_snapshot.is_empty() {
            return;
        }
        self.queues.pending_turn_input_snapshot.clear();
        self.queues.suppressed_preview_input_ids.clear();
        self.dirty = true;
    }

    pub fn remove_pending_turn_inputs(&mut self, input_ids: &[String]) {
        if input_ids.is_empty() || self.queues.pending_turn_input_snapshot.is_empty() {
            return;
        }
        let before = self.queues.pending_turn_input_snapshot.len();
        self.queues
            .pending_turn_input_snapshot
            .retain(|input| !input_ids.iter().any(|id| id == &input.input_id));
        if self.queues.pending_turn_input_snapshot.len() != before {
            self.dirty = true;
        }
    }

    pub fn pending_turn_input_snapshot(&self) -> &[lash_core::PendingTurnInput] {
        &self.queues.pending_turn_input_snapshot
    }

    pub fn suppress_queue_preview_inputs<'a>(
        &mut self,
        input_ids: impl IntoIterator<Item = &'a str>,
    ) {
        let mut changed = false;
        for input_id in input_ids {
            changed |= self
                .queues
                .suppressed_preview_input_ids
                .insert(input_id.to_string());
        }
        if changed {
            self.dirty = true;
        }
    }

    pub fn queued_input_preview_suppressed(&self, input: &lash_core::PendingTurnInput) -> bool {
        self.queues
            .suppressed_preview_input_ids
            .contains(&input.input_id)
    }

    pub fn has_queued_messages(&self) -> bool {
        self.queues.pending_turn_input_snapshot.iter().any(|input| {
            if self.queued_input_preview_suppressed(input) {
                return false;
            }
            !matches!(
                input.state,
                lash_core::TurnInputState::Cancelled | lash_core::TurnInputState::Completed
            )
        })
    }

    pub fn visible_turn_inputs_for_editing(
        &self,
    ) -> impl Iterator<Item = &lash_core::PendingTurnInput> {
        self.queues
            .pending_turn_input_snapshot
            .iter()
            .filter(|input| {
                !self.queued_input_preview_suppressed(input)
                    && input.state.is_next_turn_pending()
                    && matches!(input.ingress, lash_core::TurnInputIngress::NextTurn)
            })
    }

    pub fn prepared_turn_for_pending_input(
        &self,
        pending: &lash_core::PendingTurnInput,
    ) -> Option<PreparedTurn> {
        let draft_id = pending.source_key.as_deref().map(source_key_draft_id);
        self.queues.presentation_for_input(draft_id, &pending.input)
    }

    pub fn take_prepared_turn_for_pending_input(
        &mut self,
        pending: &lash_core::PendingTurnInput,
    ) -> Option<PreparedTurn> {
        let draft_id = pending.source_key.as_deref().map(source_key_draft_id);
        if let Some(draft_id) = draft_id
            && let Some(turn) = self.take_draft_presentation_by_id(draft_id)
        {
            return Some(turn);
        }
        let content = turn_input_display_text(&pending.input);
        if let Some(turn) = self.take_matching_draft_presentation(&content) {
            return Some(turn);
        }
        if let Some(turn) = prepared_turn_from_queued_input(&pending.input) {
            return Some(turn);
        }
        None
    }

    #[cfg(test)]
    pub(crate) fn test_seed_queued_turn_snapshot(
        &mut self,
        turn: PreparedTurn,
        ingress: lash_core::TurnInputIngress,
    ) -> String {
        let input_id = format!(
            "test-ti-{}",
            self.queues.pending_turn_input_snapshot.len() + 1
        );
        let source_key = format!("host:{}", turn.draft_id);
        self.cache_draft_presentation(turn.clone());
        let state = match ingress {
            lash_core::TurnInputIngress::ActiveTurn { .. } => {
                lash_core::TurnInputState::PendingActive
            }
            lash_core::TurnInputIngress::NextTurn => lash_core::TurnInputState::DeferredNextTurn,
        };
        self.queues
            .pending_turn_input_snapshot
            .push(lash_core::PendingTurnInput {
                input_id: input_id.clone(),
                session_id: self.session_id.clone(),
                enqueue_seq: self.queues.pending_turn_input_snapshot.len() as u64 + 1,
                source_key: Some(source_key),
                ingress,
                state,
                enqueued_at_ms: 0,
                input: crate::turn_runner::make_turn_input(&turn),
            });
        input_id
    }
}
