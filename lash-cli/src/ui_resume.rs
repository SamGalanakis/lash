use lash::Store;

use crate::app::UiResumeState;

pub(crate) fn save_ui_resume_state(store: &Store, ui_state: &UiResumeState) {
    store.save_ui_resume_state(ui_state);
}

pub(crate) fn load_ui_resume_state(store: &Store) -> UiResumeState {
    store.load_ui_resume_state().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_load_ui_resume_state_round_trip() {
        let store = Store::memory().expect("store");
        let ui_state = UiResumeState {
            streaming_output: vec!["tail".to_string()],
            streaming_output_hidden: 2,
            ..UiResumeState::default()
        };

        save_ui_resume_state(&store, &ui_state);
        let loaded = load_ui_resume_state(&store);

        assert_eq!(loaded.streaming_output_hidden, 2);
        assert_eq!(loaded.streaming_output[0], "tail");
    }
}
