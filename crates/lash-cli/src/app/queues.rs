use super::*;

fn source_key_draft_id(source_key: &str) -> &str {
    source_key
        .strip_prefix("host:")
        .or_else(|| source_key.strip_prefix("injection:"))
        .unwrap_or(source_key)
}

fn queued_work_signature(batches: &[QueuedWorkBatch]) -> Vec<(String, Option<String>, usize)> {
    batches
        .iter()
        .map(|batch| {
            (
                batch.batch_id.clone(),
                batch.source_key.clone(),
                batch.items.len(),
            )
        })
        .collect()
}

fn queued_work_batch_ids(batches: &[QueuedWorkBatch]) -> std::collections::HashSet<String> {
    batches.iter().map(|batch| batch.batch_id.clone()).collect()
}

fn batch_has_turn_input(batch: &QueuedWorkBatch) -> bool {
    batch
        .items
        .iter()
        .any(|item| matches!(item.payload, QueuedWorkPayload::TurnInput { .. }))
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

    pub fn set_queued_work_snapshot(&mut self, batches: Vec<QueuedWorkBatch>) {
        let batches = batches
            .into_iter()
            .filter(batch_has_turn_input)
            .collect::<Vec<_>>();
        let visible_batch_ids = queued_work_batch_ids(&batches);
        let suppressed_before = self.queues.suppressed_preview_batch_ids.len();
        self.queues
            .suppressed_preview_batch_ids
            .retain(|batch_id| visible_batch_ids.contains(batch_id));
        let suppressed_changed =
            self.queues.suppressed_preview_batch_ids.len() != suppressed_before;
        if queued_work_signature(&self.queues.queued_work_snapshot)
            == queued_work_signature(&batches)
        {
            if suppressed_changed {
                self.dirty = true;
            }
            return;
        }
        self.queues.queued_work_snapshot = batches;
        self.dirty = true;
    }

    pub fn clear_queued_work_snapshot(&mut self) {
        if self.queues.queued_work_snapshot.is_empty() {
            return;
        }
        self.queues.queued_work_snapshot.clear();
        self.queues.suppressed_preview_batch_ids.clear();
        self.dirty = true;
    }

    pub fn remove_queued_work_batches(&mut self, batch_ids: &[String]) {
        if batch_ids.is_empty() || self.queues.queued_work_snapshot.is_empty() {
            return;
        }
        let before = self.queues.queued_work_snapshot.len();
        self.queues
            .queued_work_snapshot
            .retain(|batch| !batch_ids.iter().any(|id| id == &batch.batch_id));
        if self.queues.queued_work_snapshot.len() != before {
            self.dirty = true;
        }
    }

    pub fn queued_work_snapshot(&self) -> &[QueuedWorkBatch] {
        &self.queues.queued_work_snapshot
    }

    pub fn suppress_queue_preview_batches<'a>(
        &mut self,
        batch_ids: impl IntoIterator<Item = &'a str>,
    ) {
        let mut changed = false;
        for batch_id in batch_ids {
            changed |= self
                .queues
                .suppressed_preview_batch_ids
                .insert(batch_id.to_string());
        }
        if changed {
            self.dirty = true;
        }
    }

    pub fn queued_batch_preview_suppressed(&self, batch: &QueuedWorkBatch) -> bool {
        self.queues
            .suppressed_preview_batch_ids
            .contains(&batch.batch_id)
    }

    pub fn has_queued_messages(&self) -> bool {
        self.queues.queued_work_snapshot.iter().any(|batch| {
            if self.queued_batch_preview_suppressed(batch) {
                return false;
            }
            batch
                .items
                .iter()
                .any(|item| matches!(item.payload, QueuedWorkPayload::TurnInput { .. }))
        })
    }

    pub fn visible_turn_batches_for_editing(&self) -> impl Iterator<Item = &QueuedWorkBatch> {
        self.queues.queued_work_snapshot.iter().filter(|batch| {
            !self.queued_batch_preview_suppressed(batch)
                && batch.slot_policy == SlotPolicy::Exclusive
                && matches!(
                    batch.delivery_policy,
                    DeliveryPolicy::EarliestSafeBoundary | DeliveryPolicy::AfterCurrentTurnCommit
                )
                && batch
                    .items
                    .iter()
                    .any(|item| matches!(item.payload, QueuedWorkPayload::TurnInput { .. }))
        })
    }

    pub fn prepared_turn_for_queued_batch(&self, batch: &QueuedWorkBatch) -> Option<PreparedTurn> {
        let draft_id = batch.source_key.as_deref().map(source_key_draft_id);
        batch.items.iter().find_map(|item| {
            let QueuedWorkPayload::TurnInput { input } = &item.payload else {
                return None;
            };
            self.queues.presentation_for_input(draft_id, input)
        })
    }

    pub fn take_prepared_turn_for_queued_batch(
        &mut self,
        batch: &QueuedWorkBatch,
    ) -> Option<PreparedTurn> {
        let draft_id = batch.source_key.as_deref().map(source_key_draft_id);
        for item in &batch.items {
            let QueuedWorkPayload::TurnInput { input } = &item.payload else {
                continue;
            };
            if let Some(draft_id) = draft_id
                && let Some(turn) = self.take_draft_presentation_by_id(draft_id)
            {
                return Some(turn);
            }
            let content = turn_input_display_text(input);
            if let Some(turn) = self.take_matching_draft_presentation(&content) {
                return Some(turn);
            }
            if let Some(turn) = prepared_turn_from_queued_input(input) {
                return Some(turn);
            }
        }
        None
    }

    #[cfg(test)]
    pub(crate) fn test_seed_queued_turn_snapshot(
        &mut self,
        turn: PreparedTurn,
        delivery_policy: DeliveryPolicy,
        slot_policy: SlotPolicy,
    ) -> String {
        let batch_id = format!("test-qwb-{}", self.queues.queued_work_snapshot.len() + 1);
        let item_id = format!("{batch_id}:item:0");
        let source_key = format!("host:{}", turn.draft_id);
        self.cache_draft_presentation(turn.clone());
        self.queues.queued_work_snapshot.push(QueuedWorkBatch {
            batch_id: batch_id.clone(),
            session_id: self.session_id.clone(),
            enqueue_seq: self.queues.queued_work_snapshot.len() as u64 + 1,
            source_key: Some(source_key),
            delivery_policy,
            slot_policy,
            merge_key: lash_core::runtime::MergeKey::Never,
            available_at_ms: 0,
            enqueued_at_ms: 0,
            items: vec![lash_core::runtime::QueuedWorkItem {
                item_id,
                payload: QueuedWorkPayload::turn_input(crate::turn_runner::make_turn_input(&turn)),
            }],
        });
        batch_id
    }
}
