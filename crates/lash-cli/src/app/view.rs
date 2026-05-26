use super::*;

impl App {
    pub fn clear(&mut self) {
        self.timeline = vec![UiTimelineItem::Splash].into();
        self.scroll_offset = 0;
        self.follow_mode = FollowOutputMode::Bottom;
        self.live_assistant.clear();
        self.live_reasoning.clear();
        self.clear_status();
        self.editor.pending_images.clear();
        self.editor.pending_large_pastes.clear();
        self.clear_live_tool_output();
        self.pending_steers.clear();
        self.queued_turns.clear();
        self.pending_process_wakes.clear();
        self.in_flight_process_wakes.clear();
        self.overlay = None;
        self.activity_state.reset();
        self.token_usage = TokenUsage::default();
        self.last_response_usage = TokenUsage::default();
        self.last_prompt_usage = None;
        self.live_output_chars_estimate = 0;
        self.live_output_tokens_estimate = 0;
        self.model_variant = None;
        self.clear_mode_indicators();
        self.plan_dock = None;
        self.processes.clear();
        self.invalidate_height_cache();
    }

    pub fn restore_prepared_turn(&mut self, turn: PreparedTurn) {
        self.editor
            .restore_turn(turn.display_text, turn.images, turn.large_pastes);
        self.update_suggestions();
    }

    pub fn next_image_marker_id(&self) -> usize {
        self.editor.next_image_marker_id()
    }

    #[cfg(test)]
    pub fn add_pending_image(&mut self, png_bytes: Vec<u8>) -> usize {
        let id = self.next_image_marker_id();
        self.editor
            .pending_images
            .push(PendingImage { id, png_bytes });
        id
    }

    pub fn begin_pending_image(&mut self, id: usize) {
        self.editor.begin_pending_image(id);
    }

    pub fn has_pending_image_jobs(&self) -> bool {
        self.editor.has_pending_image_jobs()
    }

    pub fn complete_pending_image(&mut self, id: usize, png_bytes: Vec<u8>) -> bool {
        self.editor.complete_pending_image(id, png_bytes)
    }

    pub fn fail_pending_image(&mut self, id: usize) -> bool {
        self.editor.fail_pending_image(id)
    }

    pub fn cycle_expand(&mut self) {
        let new_level = if self.expand_level == 0 { 1 } else { 0 };
        self.set_expand_level(new_level);
    }

    pub fn toggle_full_expand(&mut self) {
        let new_level = if self.expand_level != 2 { 2 } else { 1 };
        self.set_expand_level(new_level);
    }

    pub fn set_expand_level(&mut self, level: u8) {
        if self.follows_output() {
            self.expand_level = level;
            self.invalidate_height_cache();
            if self.height_cache_width > 0 && self.height_cache_vh > 0 {
                self.ensure_height_cache(self.height_cache_width, self.height_cache_vh);
                self.scroll_offset =
                    self.follow_output_anchor_offset(self.height_cache_width, self.height_cache_vh);
            } else {
                self.scroll_offset = usize::MAX;
            }
            return;
        }

        if self.height_cache_width > 0 {
            let w = self.height_cache_width;
            let vh = self.height_cache_vh;
            self.ensure_height_cache(w, vh);
        }

        let anchor = if self.height_cache_width == 0 || self.height_cache.is_empty() {
            None
        } else {
            let cache = &self.height_cache;
            let idx = cache.partition_point(|&cum| cum <= self.scroll_offset);
            if idx >= self.timeline.len() {
                None
            } else {
                let block_start = if idx == 0 { 0 } else { cache[idx - 1] };
                let skip = self.scroll_offset - block_start;
                Some((idx, skip))
            }
        };

        self.expand_level = level;
        self.invalidate_height_cache();

        if let Some((anchor_idx, anchor_skip)) = anchor {
            let w = self.height_cache_width;
            let vh = self.height_cache_vh;
            self.ensure_height_cache(w, vh);

            let new_block_start = if anchor_idx == 0 {
                0
            } else {
                self.height_cache[anchor_idx - 1]
            };
            let new_block_height = self.height_cache[anchor_idx] - new_block_start;
            let clamped_skip = anchor_skip.min(new_block_height.saturating_sub(1));
            self.scroll_offset = new_block_start + clamped_skip;
        }
    }

    pub fn scroll_up(&mut self, amount: usize) {
        if self.follows_output() {
            if self.height_cache_width > 0 && self.height_cache_vh > 0 {
                self.scroll_offset =
                    self.follow_output_anchor_offset(self.height_cache_width, self.height_cache_vh);
            } else {
                self.scroll_offset = 0;
            }
            self.follow_mode = FollowOutputMode::Paused;
        }
        self.scroll_offset = self.scroll_offset.saturating_sub(amount);
        self.follow_mode = FollowOutputMode::Paused;
    }

    pub fn scroll_down(&mut self, amount: usize, viewport_height: usize, viewport_width: usize) {
        let total = self.total_content_height(viewport_width, viewport_height);
        let max_scroll = total.saturating_sub(viewport_height);
        self.scroll_offset = self.scroll_offset.saturating_add(amount).min(max_scroll);
        if self.scroll_offset >= max_scroll {
            self.follow_mode = FollowOutputMode::Bottom;
        }
    }

    pub fn scroll_to_bottom(&mut self) {
        if !self.follows_output() {
            return;
        }
        self.scroll_offset = usize::MAX;
    }

    pub fn resume_follow_output(&mut self) {
        self.follow_mode = FollowOutputMode::Bottom;
        self.scroll_offset = usize::MAX;
    }

    pub fn resume_contextual_follow_output(&mut self) {
        self.follow_mode = FollowOutputMode::Contextual;
        self.scroll_offset = usize::MAX;
    }

    pub fn refresh_follow_output_anchor(&mut self, viewport_width: usize, viewport_height: usize) {
        if !self.follows_output() {
            return;
        }
        self.scroll_offset = self.follow_output_anchor_offset(viewport_width, viewport_height);
    }

    pub fn keep_latest_user_block_visible(&mut self) {
        if self.follow_mode != FollowOutputMode::Contextual {
            return;
        }

        let Some(last_idx) = self
            .timeline
            .iter()
            .rposition(|block| matches!(block, UiTimelineItem::UserInput(_)))
        else {
            self.scroll_to_bottom();
            return;
        };

        if self.height_cache_width == 0 || self.height_cache_vh == 0 {
            self.scroll_to_bottom();
            return;
        }

        let width = self.height_cache_width;
        let viewport_height = self.height_cache_vh;
        self.ensure_height_cache(width, viewport_height);

        let total_height = self.total_content_height(width, viewport_height);
        let max_scroll = total_height.saturating_sub(viewport_height);
        let block_start = self.block_start_offset(last_idx);
        let block_end = self.height_cache[last_idx];
        let block_height = block_end.saturating_sub(block_start);
        let block_content_start = self.block_content_start_offset(last_idx);
        let has_splash_before = self.timeline[..last_idx]
            .iter()
            .any(|block| matches!(block, UiTimelineItem::Splash));

        let awaiting_first_visible_output = self
            .live_turn
            .as_ref()
            .is_some_and(|turn| !turn.has_visible_output);

        self.scroll_offset = if awaiting_first_visible_output
            && (has_splash_before || block_height >= viewport_height)
        {
            self.contextual_follow_offset(block_content_start, max_scroll)
        } else {
            block_end.saturating_sub(viewport_height).min(max_scroll)
        };
    }

    fn follow_output_anchor_offset(
        &mut self,
        viewport_width: usize,
        viewport_height: usize,
    ) -> usize {
        let total_height = self.total_content_height(viewport_width, viewport_height);
        let max_scroll = total_height.saturating_sub(viewport_height);

        match self.follow_mode {
            FollowOutputMode::Paused => return self.scroll_offset.min(max_scroll),
            FollowOutputMode::Bottom => return max_scroll,
            FollowOutputMode::Contextual => {}
        }

        if !self.running {
            return max_scroll;
        }

        let awaiting_first_visible_output = self
            .live_turn
            .as_ref()
            .is_some_and(|turn| !turn.has_visible_output);

        if awaiting_first_visible_output {
            return self.latest_user_block_anchor_offset(max_scroll);
        }

        let anchor_output_start = self
            .live_turn
            .as_ref()
            .is_some_and(|turn| turn.output_start_anchor_pending);

        if anchor_output_start {
            if let Some(turn) = self.live_turn.as_mut() {
                turn.output_start_anchor_pending = false;
            }
            self.follow_mode = FollowOutputMode::Bottom;

            let Some(output_start) = self.latest_turn_output_start_offset() else {
                return max_scroll;
            };

            return self.contextual_follow_offset(output_start, max_scroll);
        }

        max_scroll
    }

    pub(super) fn latest_turn_output_start_offset(&self) -> Option<usize> {
        let search_start = self
            .timeline
            .iter()
            .rposition(|block| matches!(block, UiTimelineItem::UserInput(_)))
            .map(|idx| idx + 1)
            .unwrap_or(0);

        if let Some(idx) = self.timeline[search_start..]
            .iter()
            .position(Self::is_turn_visible_output_block)
            .map(|offset| search_start + offset)
        {
            return Some(self.block_content_start_offset(idx));
        }

        let history_tail = self.height_cache.last().copied().unwrap_or(0);
        if self.live_tool_output.height() > 0
            && self.live_tool_output.title.is_some()
            && self.live_tool_output_anchor_block_index().is_none()
        {
            return Some(history_tail);
        }
        if self.live_reasoning.has_renderable_output() {
            return Some(history_tail + self.live_reasoning_leading_padding());
        }
        self.live_assistant
            .has_renderable_output()
            .then_some(history_tail + self.live_assistant_leading_padding())
    }

    fn is_turn_visible_output_block(block: &UiTimelineItem) -> bool {
        matches!(
            block,
            UiTimelineItem::AssistantText(_)
                | UiTimelineItem::AssistantReasoning(_)
                | UiTimelineItem::Activity(_)
                | UiTimelineItem::ShellOutput { .. }
                | UiTimelineItem::Error(_)
                | UiTimelineItem::PluginPanel(_)
        )
    }

    fn latest_user_block_anchor_offset(&self, max_scroll: usize) -> usize {
        let Some(last_idx) = self
            .timeline
            .iter()
            .rposition(|block| matches!(block, UiTimelineItem::UserInput(_)))
        else {
            return max_scroll;
        };

        self.contextual_follow_offset(self.block_content_start_offset(last_idx), max_scroll)
    }

    pub(super) fn contextual_follow_offset(
        &self,
        content_start: usize,
        max_scroll: usize,
    ) -> usize {
        content_start
            .saturating_sub(FOLLOW_OUTPUT_CONTEXT_LINES)
            .min(max_scroll)
    }

    fn follows_output(&self) -> bool {
        self.follow_mode != FollowOutputMode::Paused
    }

    fn block_start_offset(&self, idx: usize) -> usize {
        if idx == 0 {
            0
        } else {
            self.height_cache[idx - 1]
        }
    }

    pub(super) fn block_content_start_offset(&self, idx: usize) -> usize {
        self.block_start_offset(idx) + self.block_leading_padding(idx)
    }

    fn block_leading_padding(&self, idx: usize) -> usize {
        if idx == 0 {
            return 0;
        }

        match self.timeline.get(idx) {
            Some(UiTimelineItem::UserInput(_)) => {
                usize::from(!matches!(self.timeline[idx - 1], UiTimelineItem::Splash))
            }
            Some(UiTimelineItem::AssistantText(_)) => usize::from(!matches!(
                self.timeline[idx - 1],
                UiTimelineItem::AssistantText(_) | UiTimelineItem::Splash
            )),
            _ => 0,
        }
    }

    pub fn ensure_height_cache_pub(&mut self, width: usize, viewport_height: usize) {
        self.ensure_height_cache(width, viewport_height);
        self.ensure_live_markdown_rendered(width);
    }

    pub fn height_cache_snapshot(&self) -> &[usize] {
        &self.height_cache
    }

    fn ensure_height_cache(&mut self, width: usize, viewport_height: usize) {
        let dimensions_changed =
            self.height_cache_width != width || self.height_cache_vh != viewport_height;
        if dimensions_changed {
            self.height_cache.clear();
            self.height_cache_dirty_from = 0;
        }
        if !self.height_cache.is_empty()
            && !dimensions_changed
            && self.height_cache_dirty_from >= self.timeline.len()
        {
            return;
        }
        self.height_cache_width = width;
        self.height_cache_vh = viewport_height;

        let target_len = self.timeline.len();
        if self.height_cache.len() > target_len {
            self.height_cache.truncate(target_len);
        }
        let dirty_from = self.height_cache_dirty_from.min(target_len);
        if dirty_from == 0 {
            self.height_cache.clear();
            self.height_cache.reserve(target_len);
        } else {
            self.height_cache.truncate(dirty_from);
        }
        let mut cumulative = if dirty_from == 0 {
            0
        } else {
            self.height_cache[dirty_from - 1]
        };
        for i in dirty_from..target_len {
            cumulative += self.rendered_block_height_cached(i, width, viewport_height);
            self.height_cache.push(cumulative);
        }
        self.height_cache_dirty_from = target_len;
    }

    pub fn total_content_height(&mut self, width: usize, viewport_height: usize) -> usize {
        self.ensure_height_cache(width, viewport_height);
        self.ensure_live_markdown_rendered(width);
        self.height_cache.last().copied().unwrap_or(0)
            + crate::render::live_tool_output_standalone_height(self, width)
            + self.live_reasoning_height()
            + self.live_assistant_height()
            + crate::render::plan_dock_trailing_height(self)
            + crate::render::process_trailing_height(self)
    }
}
