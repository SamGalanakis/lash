#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SelectionState {
    pub selected: Option<usize>,
}

impl SelectionState {
    pub const fn new(selected: Option<usize>) -> Self {
        Self { selected }
    }

    pub fn select(&mut self, selected: Option<usize>) {
        self.selected = selected;
    }

    pub fn clamp(&mut self, item_count: usize) {
        self.selected = match (self.selected, item_count) {
            (_, 0) => None,
            (Some(index), _) => Some(index.min(item_count - 1)),
            (None, _) => None,
        };
    }

    pub fn move_next(&mut self, item_count: usize) {
        self.selected = match (self.selected, item_count) {
            (_, 0) => None,
            (None, _) => Some(0),
            (Some(index), _) => Some((index + 1).min(item_count - 1)),
        };
    }

    pub fn move_prev(&mut self, item_count: usize) {
        self.selected = match (self.selected, item_count) {
            (_, 0) => None,
            (None, _) => Some(item_count - 1),
            (Some(index), _) => Some(index.saturating_sub(1)),
        };
    }

    pub fn home(&mut self, item_count: usize) {
        self.selected = (item_count > 0).then_some(0);
    }

    pub fn end(&mut self, item_count: usize) {
        self.selected = (item_count > 0).then_some(item_count - 1);
    }
}

#[cfg(test)]
mod tests {
    use super::SelectionState;

    #[test]
    fn selection_state_moves_and_clamps() {
        let mut state = SelectionState::default();
        state.move_next(3);
        assert_eq!(state.selected, Some(0));
        state.move_next(3);
        assert_eq!(state.selected, Some(1));
        state.move_prev(3);
        assert_eq!(state.selected, Some(0));
        state.select(Some(9));
        state.clamp(3);
        assert_eq!(state.selected, Some(2));
        state.clamp(0);
        assert_eq!(state.selected, None);
    }

    #[test]
    fn selection_state_home_end_work() {
        let mut state = SelectionState::default();
        state.end(4);
        assert_eq!(state.selected, Some(3));
        state.home(4);
        assert_eq!(state.selected, Some(0));
    }
}
