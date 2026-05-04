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

    pub fn move_cursor_word_left(&mut self) {
        self.editor.move_cursor_word_left();
    }

    pub fn move_cursor_word_right(&mut self) {
        self.editor.move_cursor_word_right();
    }

    pub fn move_cursor_home(&mut self) {
        self.editor.move_cursor_home();
    }

    pub fn move_cursor_end(&mut self) {
        self.editor.move_cursor_end();
    }

    pub fn clear_input_selection(&mut self) {
        self.editor.clear_selection();
    }

    pub fn has_input_selection(&self) -> bool {
        self.editor.has_selection()
    }

    pub fn input_selection_active(&self) -> bool {
        self.editor.selection_is_active()
    }

    pub fn input_selection_range(&self) -> Option<std::ops::Range<usize>> {
        self.editor.selected_range()
    }

    pub fn selected_input_text(&self) -> Option<String> {
        self.editor.selected_text()
    }

    pub fn start_input_selection(&mut self, offset: usize) {
        self.editor.start_selection(offset);
    }

    pub fn update_input_selection(&mut self, offset: usize) {
        self.editor.update_selection(offset);
    }

    pub fn finish_input_selection(&mut self) {
        self.editor.finish_selection();
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
        self.editor.update_suggestions(
            &self.skills,
            self.ui_extensions.as_ref(),
            &self.plugin_commands,
            self.file_index.as_ref(),
        );
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
        self.editor.complete_suggestion(
            &self.skills,
            self.ui_extensions.as_ref(),
            &self.plugin_commands,
        );
    }

    pub fn ui_extensions(&self) -> &UiExtensions {
        self.ui_extensions.as_ref()
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

    pub fn has_tree(&self) -> bool {
        matches!(&self.overlay, Some(OverlayState::Tree(state)) if !state.is_empty())
    }

    pub fn tree_up(&mut self) {
        if let Some(OverlayState::Tree(state)) = &mut self.overlay {
            state.up();
        }
    }

    pub fn tree_down(&mut self) {
        if let Some(OverlayState::Tree(state)) = &mut self.overlay {
            state.down();
        }
    }

    pub fn tree_prev_branch(&mut self) {
        if let Some(OverlayState::Tree(state)) = &mut self.overlay {
            state.collapse_or_jump_prev_branch();
        }
    }

    pub fn tree_next_branch(&mut self) {
        if let Some(OverlayState::Tree(state)) = &mut self.overlay {
            state.expand_or_jump_next_branch();
        }
    }

    pub fn take_tree_pick(&mut self) -> Option<crate::overlay::TreeSelection> {
        match self.overlay.take() {
            Some(OverlayState::Tree(mut state)) => state.take_selected(),
            other => {
                self.overlay = other;
                None
            }
        }
    }

    pub fn dismiss_tree(&mut self) {
        if matches!(self.overlay, Some(OverlayState::Tree(_))) {
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

    pub fn prompt_has_options(&self) -> bool {
        match &self.overlay {
            Some(OverlayState::Prompt(prompt)) => prompt.has_options(),
            _ => false,
        }
    }

    pub fn prompt_uses_split_layout(&self) -> bool {
        match &self.overlay {
            Some(OverlayState::Prompt(prompt)) => prompt.uses_split_layout(),
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

    pub fn show_tree(&mut self, roots: Vec<lash::SessionMessageTreeNode>) {
        self.overlay = Some(OverlayState::Tree(crate::overlay::TreeState::new(roots)));
    }

    pub fn tree_state(&self) -> Option<&crate::overlay::TreeState> {
        match &self.overlay {
            Some(OverlayState::Tree(state)) => Some(state),
            _ => None,
        }
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

    pub fn prompt_state(&self) -> Option<&PromptState> {
        match &self.overlay {
            Some(OverlayState::Prompt(prompt)) => Some(prompt),
            _ => None,
        }
    }

    pub fn input(&self) -> &str {
        &self.editor.input
    }

    /// True when the editor has pending images or large-paste payloads
    /// attached, even if `input()` is empty. Used by the input render
    /// snapshot to decide whether to show the idle placeholder hint.
    pub fn has_pending_input_payload(&self) -> bool {
        !self.editor.pending_images.is_empty() || !self.editor.pending_large_pastes.is_empty()
    }

    pub fn set_input(&mut self, input: String) {
        // Wholesale replacement (tree-selection seeding, skill
        // prompt staging, test setup) — not a user edit, so drop
        // the undo history rather than recording it.
        self.editor.input = input;
        self.editor.cursor_pos = self.editor.input.len();
        self.editor.clear_undo_history_from_app();
    }

    pub fn cursor_pos(&self) -> usize {
        self.editor.cursor_pos
    }

    /// Undo the most recent edit to the input draft. Bound to
    /// `Ctrl+Z` in interactive mode.
    pub fn editor_undo(&mut self) -> bool {
        self.editor.undo()
    }

    /// Redo an edit that was previously undone. Bound to `Ctrl+Y`.
    pub fn editor_redo(&mut self) -> bool {
        self.editor.redo()
    }

    pub fn suggestions(&self) -> &[crate::editor::Suggestion] {
        &self.editor.suggestions
    }

    pub fn suggestion_kind(&self) -> SuggestionKind {
        self.editor.suggestion_kind
    }

    pub fn suggestion_idx(&self) -> usize {
        self.editor.suggestion_idx
    }
}
