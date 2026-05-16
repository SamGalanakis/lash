#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScrollState {
    pub offset: usize,
}

impl ScrollState {
    pub const fn new(offset: usize) -> Self {
        Self { offset }
    }

    pub fn clamp(&mut self, content_len: usize, viewport_len: usize) {
        self.offset = self.offset.min(max_offset(content_len, viewport_len));
    }

    pub fn scroll_by(&mut self, delta: isize, content_len: usize, viewport_len: usize) {
        if delta.is_negative() {
            self.offset = self.offset.saturating_sub(delta.unsigned_abs());
        } else {
            self.offset = self.offset.saturating_add(delta as usize);
        }
        self.clamp(content_len, viewport_len);
    }

    pub fn page_down(&mut self, viewport_len: usize, content_len: usize) {
        self.scroll_by(viewport_len as isize, content_len, viewport_len);
    }

    pub fn page_up(&mut self, viewport_len: usize, content_len: usize) {
        self.scroll_by(-(viewport_len as isize), content_len, viewport_len);
    }

    pub fn to_start(&mut self) {
        self.offset = 0;
    }

    pub fn to_end(&mut self, content_len: usize, viewport_len: usize) {
        self.offset = max_offset(content_len, viewport_len);
    }

    pub fn ensure_visible(
        &mut self,
        start: usize,
        end_exclusive: usize,
        viewport_len: usize,
        content_len: usize,
    ) {
        if content_len == 0 {
            self.offset = 0;
            return;
        }

        let start = start.min(content_len.saturating_sub(1));
        let end_exclusive = end_exclusive.max(start + 1).min(content_len);
        if viewport_len == 0 {
            self.offset = start.min(content_len);
            return;
        }

        if start < self.offset {
            self.offset = start;
        } else {
            let visible_end = self.offset.saturating_add(viewport_len);
            if end_exclusive > visible_end {
                self.offset = end_exclusive.saturating_sub(viewport_len);
            }
        }
        self.clamp(content_len, viewport_len);
    }
}

fn max_offset(content_len: usize, viewport_len: usize) -> usize {
    if viewport_len == 0 {
        content_len
    } else {
        content_len.saturating_sub(viewport_len)
    }
}

#[cfg(test)]
mod tests {
    use super::ScrollState;

    #[test]
    fn scroll_state_clamps_to_content_bounds() {
        let mut state = ScrollState::new(99);
        state.clamp(10, 4);
        assert_eq!(state.offset, 6);
    }

    #[test]
    fn scroll_state_page_moves_by_viewport() {
        let mut state = ScrollState::default();
        state.page_down(5, 20);
        assert_eq!(state.offset, 5);
        state.page_up(3, 20);
        assert_eq!(state.offset, 2);
    }

    #[test]
    fn scroll_state_ensure_visible_adjusts_offset() {
        let mut state = ScrollState::new(0);
        state.ensure_visible(8, 9, 4, 12);
        assert_eq!(state.offset, 5);
        state.ensure_visible(2, 3, 4, 12);
        assert_eq!(state.offset, 2);
    }
}
