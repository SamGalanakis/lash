use crate::app::App;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum UiAction {
    InputInsertText(String),
    InputInsertPastedText(String),
    InputBackspace,
    InputDelete,
    MoveCursorLeft,
    MoveCursorRight,
    MoveCursorHome,
    MoveCursorEnd,
    HistoryUp,
    HistoryDown,
    SuggestionUp,
    SuggestionDown,
    SuggestionComplete,
    PromptUp,
    PromptDown,
    PromptScrollUp(usize),
    PromptScrollDown(usize),
    PromptToggleCurrentOption,
    PromptToggleNoteFocus,
    PromptInsertText(String),
    PromptBackspace,
    SubmitPrompt,
    DismissPrompt,
    ScrollUp(usize),
    ScrollDown(usize),
    SessionPickerUp,
    SessionPickerDown,
    SubmitSessionPicker,
    DismissSessionPicker,
    TreeUp,
    TreeDown,
    TreePrevBranch,
    TreeNextBranch,
    SubmitTree,
    DismissTree,
    SkillPickerUp,
    SkillPickerDown,
    SubmitSkillPicker,
    DismissSkillPicker,
    ClearSelection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UiActionContext {
    pub viewport_width: usize,
    pub viewport_height: usize,
    pub prompt_max_scroll: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum UiActionOutcome {
    None,
    SessionPicked(Option<String>),
    TreePicked(Option<crate::overlay::TreeSelection>),
    SkillPicked(Option<String>),
}

pub(crate) fn apply_ui_action(
    app: &mut App,
    action: UiAction,
    context: UiActionContext,
) -> UiActionOutcome {
    match action {
        UiAction::InputInsertText(text) => {
            app.insert_text(&text);
            app.update_suggestions();
            UiActionOutcome::None
        }
        UiAction::InputInsertPastedText(text) => {
            app.insert_pasted_text(&text);
            app.update_suggestions();
            UiActionOutcome::None
        }
        UiAction::InputBackspace => {
            app.backspace();
            app.update_suggestions();
            UiActionOutcome::None
        }
        UiAction::InputDelete => {
            app.delete();
            app.update_suggestions();
            UiActionOutcome::None
        }
        UiAction::MoveCursorLeft => {
            app.move_cursor_left();
            UiActionOutcome::None
        }
        UiAction::MoveCursorRight => {
            app.move_cursor_right();
            UiActionOutcome::None
        }
        UiAction::MoveCursorHome => {
            app.move_cursor_home();
            UiActionOutcome::None
        }
        UiAction::MoveCursorEnd => {
            app.move_cursor_end();
            UiActionOutcome::None
        }
        UiAction::HistoryUp => {
            app.history_up();
            UiActionOutcome::None
        }
        UiAction::HistoryDown => {
            app.history_down();
            UiActionOutcome::None
        }
        UiAction::SuggestionUp => {
            app.suggestion_up();
            UiActionOutcome::None
        }
        UiAction::SuggestionDown => {
            app.suggestion_down();
            UiActionOutcome::None
        }
        UiAction::SuggestionComplete => {
            app.complete_suggestion();
            app.update_suggestions();
            UiActionOutcome::None
        }
        UiAction::PromptUp => {
            app.prompt_up();
            UiActionOutcome::None
        }
        UiAction::PromptDown => {
            app.prompt_down();
            UiActionOutcome::None
        }
        UiAction::PromptScrollUp(amount) => {
            app.prompt_scroll_up(amount);
            UiActionOutcome::None
        }
        UiAction::PromptScrollDown(amount) => {
            app.prompt_scroll_down(amount, context.prompt_max_scroll);
            UiActionOutcome::None
        }
        UiAction::PromptToggleCurrentOption => {
            app.prompt_toggle_current_option();
            UiActionOutcome::None
        }
        UiAction::PromptToggleNoteFocus => {
            app.prompt_toggle_note_focus();
            UiActionOutcome::None
        }
        UiAction::PromptInsertText(text) => {
            app.prompt_insert_text(&text);
            UiActionOutcome::None
        }
        UiAction::PromptBackspace => {
            app.prompt_backspace();
            UiActionOutcome::None
        }
        UiAction::SubmitPrompt => {
            let _ = app.take_prompt_response();
            UiActionOutcome::None
        }
        UiAction::DismissPrompt => {
            app.dismiss_prompt();
            UiActionOutcome::None
        }
        UiAction::ScrollUp(amount) => {
            app.scroll_up(amount);
            UiActionOutcome::None
        }
        UiAction::ScrollDown(amount) => {
            app.scroll_down(amount, context.viewport_height, context.viewport_width);
            UiActionOutcome::None
        }
        UiAction::SessionPickerUp => {
            app.session_picker_up();
            UiActionOutcome::None
        }
        UiAction::SessionPickerDown => {
            app.session_picker_down();
            UiActionOutcome::None
        }
        UiAction::SubmitSessionPicker => UiActionOutcome::SessionPicked(app.take_session_pick()),
        UiAction::DismissSessionPicker => {
            app.dismiss_session_picker();
            UiActionOutcome::None
        }
        UiAction::TreeUp => {
            app.tree_up();
            UiActionOutcome::None
        }
        UiAction::TreeDown => {
            app.tree_down();
            UiActionOutcome::None
        }
        UiAction::TreePrevBranch => {
            app.tree_prev_branch();
            UiActionOutcome::None
        }
        UiAction::TreeNextBranch => {
            app.tree_next_branch();
            UiActionOutcome::None
        }
        UiAction::SubmitTree => UiActionOutcome::TreePicked(app.take_tree_pick()),
        UiAction::DismissTree => {
            app.dismiss_tree();
            UiActionOutcome::None
        }
        UiAction::SkillPickerUp => {
            app.skill_picker_up();
            UiActionOutcome::None
        }
        UiAction::SkillPickerDown => {
            app.skill_picker_down();
            UiActionOutcome::None
        }
        UiAction::SubmitSkillPicker => UiActionOutcome::SkillPicked(app.take_skill_pick()),
        UiAction::DismissSkillPicker => {
            app.dismiss_skill_picker();
            UiActionOutcome::None
        }
        UiAction::ClearSelection => {
            app.clear_selection();
            UiActionOutcome::None
        }
    }
}
