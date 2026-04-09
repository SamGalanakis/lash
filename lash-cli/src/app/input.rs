use super::*;

impl App {
    /// Navigate input history with up arrow.
    /// In multi-line mode, if cursor is not on the first line, moves cursor up instead.
    pub fn history_up(&mut self) {
        if self.editor.input.is_empty()
            && self.editor.input_history_idx.is_none()
            && let Some((turn, _was_pending)) = self.take_last_queued_turn()
        {
            self.restore_prepared_turn(turn);
            return;
        }
        self.editor.history_up();
    }

    /// Navigate input history with down arrow.
    /// In multi-line mode, if cursor is not on the last line, moves cursor down instead.
    pub fn history_down(&mut self) {
        self.editor.history_down();
    }

    /// Insert a character at cursor position.
    pub fn insert_char(&mut self, c: char) {
        self.editor.insert_char(c);
    }

    /// Insert literal text at the cursor position.
    pub fn insert_text(&mut self, text: &str) {
        self.editor.insert_text(text);
    }

    pub fn insert_pasted_text(&mut self, text: &str) {
        self.editor.insert_pasted_text(text);
    }

    /// Delete character before cursor.
    pub fn backspace(&mut self) {
        self.editor.backspace();
    }

    /// Delete character at cursor.
    pub fn delete(&mut self) {
        self.editor.delete();
    }

    pub fn move_cursor_left(&mut self) {
        self.editor.move_cursor_left();
    }

    pub fn move_cursor_right(&mut self) {
        self.editor.move_cursor_right();
    }

    pub fn move_cursor_home(&mut self) {
        self.editor.move_cursor_home();
    }

    pub fn move_cursor_end(&mut self) {
        self.editor.move_cursor_end();
    }

    /// Load input history from $LASH_HOME/history.
    pub fn load_history(&mut self) {
        self.editor.load_history();
    }

    /// Save input history to $LASH_HOME/history (last 500 entries).
    pub fn save_history(&self) {
        self.editor.save_history();
    }

    /// Update the suggestion list based on current input.
    pub fn update_suggestions(&mut self) {
        self.editor
            .update_suggestions(&self.skills, self.ui_extensions.as_ref());
    }

    /// Whether the suggestion popup is active.
    pub fn has_suggestions(&self) -> bool {
        self.editor.has_suggestions()
    }

    /// Move suggestion selection up.
    pub fn suggestion_up(&mut self) {
        self.editor.suggestion_up();
    }

    /// Move suggestion selection down.
    pub fn suggestion_down(&mut self) {
        self.editor.suggestion_down();
    }

    /// Accept the selected suggestion.
    pub fn complete_suggestion(&mut self) {
        self.editor
            .complete_suggestion(&self.skills, self.ui_extensions.as_ref());
    }

    pub fn ui_extensions(&self) -> &UiExtensions {
        self.ui_extensions.as_ref()
    }

    pub fn ui_extensions_handle(&self) -> Arc<UiExtensions> {
        Arc::clone(&self.ui_extensions)
    }

    /// Whether the session picker is active.
    pub fn has_session_picker(&self) -> bool {
        matches!(&self.overlay, Some(OverlayState::SessionPicker(state)) if !state.items.is_empty())
    }

    /// Move session picker selection up.
    pub fn session_picker_up(&mut self) {
        if let Some(OverlayState::SessionPicker(state)) = &mut self.overlay {
            state.up();
        }
    }

    /// Move session picker selection down.
    pub fn session_picker_down(&mut self) {
        if let Some(OverlayState::SessionPicker(state)) = &mut self.overlay {
            state.down();
        }
    }

    /// Get the selected session filename, clearing the picker.
    pub fn take_session_pick(&mut self) -> Option<String> {
        match self.overlay.take() {
            Some(OverlayState::SessionPicker(mut state)) => {
                state.take_selected().map(|s| s.filename)
            }
            other => {
                self.overlay = other;
                None
            }
        }
    }

    /// Dismiss the session picker without selecting.
    pub fn dismiss_session_picker(&mut self) {
        if matches!(self.overlay, Some(OverlayState::SessionPicker(_))) {
            self.overlay = None;
        }
    }

    /// Whether the skill picker is active.
    pub fn has_skill_picker(&self) -> bool {
        matches!(&self.overlay, Some(OverlayState::SkillPicker(state)) if !state.items.is_empty())
    }

    /// Move skill picker selection up.
    pub fn skill_picker_up(&mut self) {
        if let Some(OverlayState::SkillPicker(state)) = &mut self.overlay {
            state.up();
        }
    }

    /// Move skill picker selection down.
    pub fn skill_picker_down(&mut self) {
        if let Some(OverlayState::SkillPicker(state)) = &mut self.overlay {
            state.down();
        }
    }

    /// Get the selected skill name, clearing the picker.
    pub fn take_skill_pick(&mut self) -> Option<String> {
        match self.overlay.take() {
            Some(OverlayState::SkillPicker(mut state)) => {
                state.take_selected().map(|(name, _)| name)
            }
            other => {
                self.overlay = other;
                None
            }
        }
    }

    /// Dismiss the skill picker without selecting.
    pub fn dismiss_skill_picker(&mut self) {
        if matches!(self.overlay, Some(OverlayState::SkillPicker(_))) {
            self.overlay = None;
        }
    }

    /// Whether the prompt dialog is active.
    pub fn has_prompt(&self) -> bool {
        matches!(self.overlay, Some(OverlayState::Prompt(_)))
    }

    pub fn has_wait(&self) -> bool {
        matches!(self.overlay, Some(OverlayState::Wait(_)))
    }

    /// Whether the prompt is currently in text-entry mode.
    pub fn is_prompt_text_entry(&self) -> bool {
        match &self.overlay {
            Some(OverlayState::Prompt(prompt)) => prompt.is_text_entry(),
            _ => false,
        }
    }

    /// Whether the prompt supports an optional note field.
    pub fn prompt_supports_note(&self) -> bool {
        match &self.overlay {
            Some(OverlayState::Prompt(prompt)) => prompt.supports_note(),
            _ => false,
        }
    }

    /// Whether the prompt allows selecting multiple options.
    pub fn is_prompt_multi_select(&self) -> bool {
        match &self.overlay {
            Some(OverlayState::Prompt(prompt)) => prompt.is_multi(),
            _ => false,
        }
    }

    pub fn prompt_up(&mut self) {
        if let Some(OverlayState::Prompt(p)) = &mut self.overlay {
            p.move_up();
        }
    }

    pub fn prompt_down(&mut self) {
        if let Some(OverlayState::Prompt(p)) = &mut self.overlay {
            p.move_down();
        }
    }

    pub fn prompt_scroll_up(&mut self, amount: usize) {
        if let Some(OverlayState::Prompt(p)) = &mut self.overlay {
            p.scroll_up(amount);
        }
    }

    pub fn prompt_scroll_down(&mut self, amount: usize, max_scroll: usize) {
        if let Some(OverlayState::Prompt(p)) = &mut self.overlay {
            p.scroll_down(amount, max_scroll);
        }
    }

    pub fn prompt_toggle_current_option(&mut self) {
        if let Some(OverlayState::Prompt(p)) = &mut self.overlay {
            p.toggle_current();
        }
    }

    pub fn prompt_toggle_note_focus(&mut self) {
        if let Some(OverlayState::Prompt(p)) = &mut self.overlay {
            p.toggle_text_focus();
        }
    }

    pub fn prompt_insert_text(&mut self, text: &str) {
        if let Some(OverlayState::Prompt(p)) = &mut self.overlay {
            p.insert_text(text);
        }
    }

    pub fn prompt_backspace(&mut self) {
        if let Some(OverlayState::Prompt(p)) = &mut self.overlay {
            p.backspace();
        }
    }

    pub fn take_prompt_response(&mut self) -> Option<String> {
        match self.overlay.take() {
            Some(OverlayState::Prompt(p)) => {
                let response = p.submitted_response();
                let display = p.display_response(&response);
                let _ = p.response_tx.send(response);
                self.invalidate_height_cache();
                self.scroll_to_bottom();
                self.dirty = true;
                if !display.trim().is_empty() {
                    if p.request.is_freeform() {
                        self.push_prompt_response_user_block(display.clone());
                    } else {
                        self.pending_option_prompt_response = Some(display.clone());
                    }
                    return Some(display);
                }
            }
            other => {
                self.overlay = other;
            }
        }
        None
    }

    pub fn dismiss_prompt(&mut self) {
        match self.overlay.take() {
            Some(OverlayState::Prompt(p)) => {
                let _ = p.response_tx.send(p.dismissed_response());
                self.invalidate_height_cache();
                self.scroll_to_bottom();
                self.dirty = true;
            }
            other => self.overlay = other,
        }
    }

    pub fn show_session_picker(&mut self, items: Vec<crate::session_log::SessionInfo>) {
        self.overlay = Some(OverlayState::SessionPicker(PickerState::new(items)));
    }

    pub fn session_picker_state(&self) -> Option<&PickerState<crate::session_log::SessionInfo>> {
        match &self.overlay {
            Some(OverlayState::SessionPicker(state)) => Some(state),
            _ => None,
        }
    }

    pub fn show_skill_picker(&mut self, items: Vec<(String, String)>) {
        self.overlay = Some(OverlayState::SkillPicker(PickerState::new(items)));
    }

    pub fn skill_picker_state(&self) -> Option<&PickerState<(String, String)>> {
        match &self.overlay {
            Some(OverlayState::SkillPicker(state)) => Some(state),
            _ => None,
        }
    }

    pub fn show_prompt(&mut self, prompt: PromptState) {
        self.overlay = Some(OverlayState::Prompt(prompt));
    }

    pub fn show_wait(&mut self, wait: WaitState) {
        self.overlay = Some(OverlayState::Wait(wait));
        self.dirty = true;
    }

    pub fn prompt_state(&self) -> Option<&PromptState> {
        match &self.overlay {
            Some(OverlayState::Prompt(prompt)) => Some(prompt),
            _ => None,
        }
    }

    pub fn wait_state(&self) -> Option<&WaitState> {
        match &self.overlay {
            Some(OverlayState::Wait(wait)) => Some(wait),
            _ => None,
        }
    }

    pub fn wait_remaining_seconds(&self) -> Option<u64> {
        self.wait_state().map(WaitState::remaining_seconds)
    }

    pub fn wait_timed_out(&self) -> bool {
        self.wait_state().is_some_and(WaitState::timed_out)
    }

    pub fn resume_wait(&mut self) {
        match self.overlay.take() {
            Some(OverlayState::Wait(wait)) => {
                let _ = wait.response_tx.send(wait.resume_response());
                self.dirty = true;
            }
            other => self.overlay = other,
        }
    }

    pub fn skip_wait(&mut self) {
        self.resume_wait();
    }

    pub fn timeout_wait(&mut self) {
        match self.overlay.take() {
            Some(OverlayState::Wait(wait)) => {
                let _ = wait.response_tx.send(wait.timeout_response());
                self.dirty = true;
            }
            other => self.overlay = other,
        }
    }

    pub fn input(&self) -> &str {
        &self.editor.input
    }

    pub fn set_input(&mut self, input: String) {
        self.editor.input = input;
        self.editor.cursor_pos = self.editor.input.len();
    }

    pub fn cursor_pos(&self) -> usize {
        self.editor.cursor_pos
    }

    pub fn suggestions(&self) -> &[(String, String)] {
        &self.editor.suggestions
    }

    pub fn suggestion_kind(&self) -> SuggestionKind {
        self.editor.suggestion_kind
    }

    pub fn suggestion_idx(&self) -> usize {
        self.editor.suggestion_idx
    }
}
