#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FocusId(pub u64);

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FocusState {
    active: Option<FocusId>,
    stack: Vec<FocusId>,
}

impl FocusState {
    pub fn active(&self) -> Option<FocusId> {
        self.active
    }

    pub fn set(&mut self, focus: FocusId) {
        self.active = Some(focus);
    }

    pub fn clear(&mut self) {
        self.active = None;
        self.stack.clear();
    }

    pub fn push(&mut self, focus: FocusId) {
        if let Some(active) = self.active {
            self.stack.push(active);
        }
        self.active = Some(focus);
    }

    pub fn pop_restore(&mut self) -> Option<FocusId> {
        self.active = self.stack.pop();
        self.active
    }
}

#[cfg(test)]
mod tests {
    use super::{FocusId, FocusState};

    #[test]
    fn focus_set_and_clear() {
        let mut state = FocusState::default();
        state.set(FocusId(7));
        assert_eq!(state.active(), Some(FocusId(7)));
        state.clear();
        assert_eq!(state.active(), None);
    }

    #[test]
    fn overlay_focus_restores_previous_focus() {
        let mut state = FocusState::default();
        state.set(FocusId(1));
        state.push(FocusId(2));
        assert_eq!(state.active(), Some(FocusId(2)));
        assert_eq!(state.pop_restore(), Some(FocusId(1)));
        assert_eq!(state.active(), Some(FocusId(1)));
    }
}
