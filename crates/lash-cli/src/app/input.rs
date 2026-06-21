use super::*;

impl App {
    /// Navigate input history with up arrow.
    /// In multi-line mode, if cursor is not on the first line, moves cursor up instead.
    pub fn history_up(&mut self) {
        self.editor.history_up();
    }

    /// Insert a character at cursor position.
    pub fn insert_char(&mut self, c: char) {
        self.clear_process_selection();
        self.editor.insert_char(c);
    }

    /// Insert literal text at the cursor position.
    pub fn insert_text(&mut self, text: &str) {
        self.clear_process_selection();
        self.editor.insert_text(text);
    }

    pub fn insert_pasted_text(&mut self, text: &str) {
        self.clear_process_selection();
        self.editor.insert_pasted_text(text);
    }

    pub fn clear_input_selection(&mut self) {
        self.editor.clear_selection();
    }

    pub fn clear_draft(&mut self) {
        self.editor
            .restore_turn(String::new(), Vec::new(), Vec::new());
        self.update_suggestions();
        self.clear_process_selection();
        self.dirty = true;
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
            self.file_index.as_ref(),
        );
    }

    /// Whether the suggestion popup is active.
    pub fn has_suggestions(&self) -> bool {
        self.editor.has_suggestions()
    }

    pub fn dismiss_suggestions(&mut self) {
        if self.editor.has_suggestions() {
            self.editor.suggestions.clear();
            self.editor.suggestion_idx = 0;
            self.editor.suggestion_kind = SuggestionKind::None;
            self.dirty = true;
        }
    }

    /// Accept the selected suggestion.
    pub fn complete_suggestion(&mut self) {
        self.editor
            .complete_suggestion(&self.skills, self.ui_extensions.as_ref());
    }

    pub fn ui_extensions(&self) -> &TuiExtensions {
        self.ui_extensions.as_ref()
    }

    /// Whether the session picker is active.
    pub fn has_session_picker(&self) -> bool {
        matches!(&self.overlay, Some(OverlayState::SessionPicker(state)) if !state.items.is_empty())
    }

    pub fn has_command_palette(&self) -> bool {
        matches!(&self.overlay, Some(OverlayState::CommandPalette(state)) if !state.items.is_empty())
    }

    pub fn show_command_palette(&mut self, items: Vec<crate::overlay::CommandPaletteItem>) {
        self.clear_selection();
        self.clear_input_selection();
        self.overlay = Some(OverlayState::CommandPalette(
            crate::overlay::CommandPaletteState::new(items),
        ));
        self.dirty = true;
    }

    pub fn command_palette_state(&self) -> Option<&crate::overlay::CommandPaletteState> {
        match &self.overlay {
            Some(OverlayState::CommandPalette(state)) => Some(state),
            _ => None,
        }
    }

    pub fn command_palette_up(&mut self) {
        if let Some(OverlayState::CommandPalette(state)) = &mut self.overlay {
            state.up();
            self.dirty = true;
        }
    }

    pub fn command_palette_down(&mut self) {
        if let Some(OverlayState::CommandPalette(state)) = &mut self.overlay {
            state.down();
            self.dirty = true;
        }
    }

    pub fn command_palette_page_up(&mut self, amount: usize) {
        if let Some(OverlayState::CommandPalette(state)) = &mut self.overlay {
            state.page_up(amount);
            self.dirty = true;
        }
    }

    pub fn command_palette_page_down(&mut self, amount: usize) {
        if let Some(OverlayState::CommandPalette(state)) = &mut self.overlay {
            state.page_down(amount);
            self.dirty = true;
        }
    }

    pub fn command_palette_home(&mut self) {
        if let Some(OverlayState::CommandPalette(state)) = &mut self.overlay {
            state.home();
            self.dirty = true;
        }
    }

    pub fn command_palette_end(&mut self) {
        if let Some(OverlayState::CommandPalette(state)) = &mut self.overlay {
            state.end();
            self.dirty = true;
        }
    }

    pub fn command_palette_insert_query_char(&mut self, ch: char) {
        if let Some(OverlayState::CommandPalette(state)) = &mut self.overlay {
            state.push_query_char(ch);
            self.dirty = true;
        }
    }

    pub fn command_palette_backspace_query(&mut self) {
        if let Some(OverlayState::CommandPalette(state)) = &mut self.overlay {
            state.pop_query_char();
            self.dirty = true;
        }
    }

    pub fn take_command_palette_action(&mut self) -> Option<crate::overlay::CommandPaletteAction> {
        match self.overlay.take() {
            Some(OverlayState::CommandPalette(state)) => state.selected_action(),
            other => {
                self.overlay = other;
                None
            }
        }
    }

    pub fn dismiss_command_palette(&mut self) {
        if matches!(self.overlay, Some(OverlayState::CommandPalette(_))) {
            self.overlay = None;
            self.dirty = true;
        }
    }

    pub fn has_document(&self) -> bool {
        matches!(self.overlay, Some(OverlayState::Document(_)))
    }

    pub fn show_document(&mut self, document: crate::overlay::DocumentState) {
        self.clear_selection();
        self.clear_input_selection();
        self.overlay = Some(OverlayState::Document(document));
        self.dirty = true;
    }

    pub fn document_state(&self) -> Option<&crate::overlay::DocumentState> {
        match &self.overlay {
            Some(OverlayState::Document(state)) => Some(state),
            _ => None,
        }
    }

    pub fn dismiss_document(&mut self) {
        if matches!(self.overlay, Some(OverlayState::Document(_))) {
            self.overlay = None;
            self.dirty = true;
        }
    }

    pub fn document_scroll_up(&mut self, amount: usize) {
        if let Some(OverlayState::Document(state)) = &mut self.overlay {
            state.scroll_up(amount);
            self.dirty = true;
        }
    }

    pub fn document_scroll_down(&mut self, amount: usize, max_scroll: usize) {
        if let Some(OverlayState::Document(state)) = &mut self.overlay {
            state.scroll_down(amount, max_scroll);
            self.dirty = true;
        }
    }

    /// Move session picker selection up.
    pub fn session_picker_up(&mut self) {
        if let Some(OverlayState::SessionPicker(state)) = &mut self.overlay {
            state.up();
            self.dirty = true;
        }
    }

    /// Move session picker selection down.
    pub fn session_picker_down(&mut self) {
        if let Some(OverlayState::SessionPicker(state)) = &mut self.overlay {
            state.down();
            self.dirty = true;
        }
    }

    /// Get the selected session filename, clearing the picker.
    pub fn take_session_pick(&mut self) -> Option<String> {
        match self.overlay.take() {
            Some(OverlayState::SessionPicker(state)) => match state.selected_filename() {
                Some(filename) => Some(filename),
                None => {
                    self.overlay = Some(OverlayState::SessionPicker(state));
                    None
                }
            },
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
            self.dirty = true;
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

    pub fn has_process_overview(&self) -> bool {
        matches!(self.overlay, Some(OverlayState::ProcessOverview(_)))
    }

    pub fn show_process_overview(&mut self, state: crate::overlay::ProcessOverviewState) {
        self.overlay = Some(OverlayState::ProcessOverview(state));
        self.dirty = true;
    }

    pub fn process_overview_state(&self) -> Option<&crate::overlay::ProcessOverviewState> {
        match &self.overlay {
            Some(OverlayState::ProcessOverview(state)) => Some(state),
            _ => None,
        }
    }

    pub fn dismiss_process_overview(&mut self) {
        if matches!(self.overlay, Some(OverlayState::ProcessOverview(_))) {
            self.overlay = None;
            self.dirty = true;
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
                        self.queues.pending_option_prompt_response = Some(display.clone());
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
        self.overlay = Some(OverlayState::SessionPicker(
            crate::overlay::SessionPickerState::new(items),
        ));
        self.dirty = true;
    }

    pub fn session_picker_state(&self) -> Option<&crate::overlay::SessionPickerState> {
        match &self.overlay {
            Some(OverlayState::SessionPicker(state)) => Some(state),
            _ => None,
        }
    }

    pub fn session_picker_insert_query_char(&mut self, ch: char) {
        if let Some(OverlayState::SessionPicker(state)) = &mut self.overlay {
            state.push_query_char(ch);
            self.dirty = true;
        }
    }

    pub fn session_picker_backspace_query(&mut self) {
        if let Some(OverlayState::SessionPicker(state)) = &mut self.overlay {
            state.pop_query_char();
            self.dirty = true;
        }
    }

    pub fn show_skill_picker(&mut self, items: Vec<(String, String)>) {
        self.overlay = Some(OverlayState::SkillPicker(PickerState::new(items)));
    }

    pub fn show_tree(&mut self, roots: Vec<lash_core::SessionMessageTreeNode>) {
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
