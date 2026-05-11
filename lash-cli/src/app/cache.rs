use super::*;

#[derive(Clone, Debug)]
pub(super) struct BlockRenderCacheEntry {
    pub(super) width: usize,
    pub(super) viewport_height: usize,
    pub(super) expand_level: u8,
    pub(super) lines: Vec<Line<'static>>,
}

impl App {
    /// Mark the height cache as stale so it will be recomputed on next access.
    pub fn invalidate_height_cache(&mut self) {
        self.height_cache_dirty_from = 0;
        self.block_render_cache.clear();
    }

    pub(super) fn invalidate_height_cache_from(&mut self, idx: usize) {
        if self.timeline.is_empty() {
            self.height_cache.clear();
            self.height_cache_dirty_from = 0;
            self.block_render_cache.clear();
            return;
        }
        if self.height_cache.is_empty() {
            self.height_cache_dirty_from = 0;
        }
        if idx < self.block_render_cache.len() {
            self.block_render_cache.truncate(idx);
        }
        self.height_cache_dirty_from = self
            .height_cache_dirty_from
            .min(idx.min(self.timeline.len().saturating_sub(1)));
    }

    pub(super) fn live_reasoning_tail_index(&self) -> Option<usize> {
        self.running
            .then(|| self.timeline.len().checked_sub(1))
            .flatten()
            .filter(|&idx| {
                matches!(
                    self.timeline.get(idx),
                    Some(UiTimelineItem::AssistantReasoning(_))
                )
            })
    }

    pub(super) fn append_invalidation_start(&self) -> usize {
        self.live_reasoning_tail_index()
            .unwrap_or(self.timeline.len())
    }

    pub(super) fn invalidate_live_reasoning_tail(&mut self) {
        if let Some(idx) = self.live_reasoning_tail_index() {
            self.invalidate_height_cache_from(idx);
        }
    }

    pub(crate) fn rendered_block_lines_cached(
        &mut self,
        idx: usize,
        viewport_width: usize,
        viewport_height: usize,
    ) -> &[Line<'static>] {
        if self.block_render_cache.len() <= idx {
            self.block_render_cache.resize_with(idx + 1, || None);
        }

        let needs_refresh = self.block_render_cache[idx].as_ref().is_none_or(|entry| {
            entry.width != viewport_width
                || entry.viewport_height != viewport_height
                || entry.expand_level != self.expand_level
        });

        if needs_refresh {
            let lines =
                crate::render::render_block_lines(self, idx, viewport_width, viewport_height);
            self.block_render_cache[idx] = Some(BlockRenderCacheEntry {
                width: viewport_width,
                viewport_height,
                expand_level: self.expand_level,
                lines,
            });
        }

        self.block_render_cache[idx]
            .as_ref()
            .map(|entry| entry.lines.as_slice())
            .unwrap_or(&[])
    }

    pub(crate) fn rendered_block_height_cached(
        &mut self,
        idx: usize,
        viewport_width: usize,
        viewport_height: usize,
    ) -> usize {
        self.rendered_block_lines_cached(idx, viewport_width, viewport_height)
            .len()
    }
}
