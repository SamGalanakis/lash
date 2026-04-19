use std::collections::BTreeSet;

use lash::{
    Message, MessageRole, PartKind, PromptRequest, PromptResponse, PromptSelectionMode,
    SessionMessageTreeNode,
};

use crate::session_log::SessionInfo;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PromptFocus {
    #[default]
    Options,
    Text,
}

#[derive(Debug)]
pub struct PromptState {
    pub request: PromptRequest,
    pub focus: PromptFocus,
    pub cursor: usize,
    pub scroll_offset: usize,
    pub selected: BTreeSet<usize>,
    pub reply_text: String,
    pub reply_cursor: usize,
    pub response_tx: std::sync::mpsc::Sender<PromptResponse>,
}

impl PromptState {
    pub fn has_review_content(&self) -> bool {
        self.request.panel.is_some() || !self.request.question.trim().is_empty()
    }

    pub fn uses_split_layout(&self) -> bool {
        self.has_review_content() && (self.has_options() || self.shows_text_input())
    }

    pub fn has_options(&self) -> bool {
        !self.request.options.is_empty()
    }

    pub fn is_freeform(&self) -> bool {
        self.request.is_freeform()
    }

    pub fn supports_note(&self) -> bool {
        self.request.allows_note()
    }

    pub fn is_multi(&self) -> bool {
        self.has_options() && matches!(self.request.selection_mode, PromptSelectionMode::Multi)
    }

    pub fn is_text_entry(&self) -> bool {
        self.is_freeform() || (self.supports_note() && matches!(self.focus, PromptFocus::Text))
    }

    pub fn shows_text_input(&self) -> bool {
        self.is_freeform() || self.supports_note()
    }

    pub fn selected_option_idx(&self) -> Option<usize> {
        if self.has_options() && self.cursor < self.request.options.len() {
            Some(self.cursor)
        } else {
            None
        }
    }

    pub fn option_label(&self, idx: usize) -> Option<&str> {
        self.request.options.get(idx).map(String::as_str)
    }

    pub fn option_marked(&self, idx: usize) -> bool {
        if self.is_multi() {
            self.selected.contains(&idx)
        } else {
            self.selected_option_idx() == Some(idx)
        }
    }

    pub fn move_up(&mut self) {
        if self.has_options() {
            self.cursor = self.cursor.saturating_sub(1);
        }
    }

    pub fn move_down(&mut self) {
        if self.has_options() {
            self.cursor = (self.cursor + 1).min(self.request.options.len().saturating_sub(1));
        }
    }

    pub fn scroll_up(&mut self, amount: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(amount);
    }

    pub fn scroll_down(&mut self, amount: usize, max_scroll: usize) {
        self.scroll_offset = self.scroll_offset.saturating_add(amount).min(max_scroll);
    }

    pub fn toggle_current(&mut self) {
        if !self.is_multi() {
            return;
        }
        if !self.selected.insert(self.cursor) {
            self.selected.remove(&self.cursor);
        }
    }

    pub fn toggle_text_focus(&mut self) {
        if !self.supports_note() {
            return;
        }
        self.focus = match self.focus {
            PromptFocus::Options => PromptFocus::Text,
            PromptFocus::Text => PromptFocus::Options,
        };
    }

    pub fn insert_text(&mut self, text: &str) {
        self.reply_text.insert_str(self.reply_cursor, text);
        self.reply_cursor += text.len();
    }

    pub fn backspace(&mut self) {
        if self.reply_cursor == 0 {
            return;
        }
        let prev = self.reply_text[..self.reply_cursor]
            .char_indices()
            .next_back()
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.reply_text.drain(prev..self.reply_cursor);
        self.reply_cursor = prev;
    }

    fn response_note(&self) -> Option<String> {
        if !self.supports_note() || self.reply_text.trim().is_empty() {
            return None;
        }
        Some(self.reply_text.clone())
    }

    fn format_note_display(base: String, note: Option<&str>) -> String {
        let Some(note) = note.filter(|note| !note.trim().is_empty()) else {
            return base;
        };
        if base.trim().is_empty() {
            format!("Note: {note}")
        } else {
            format!("{base}\n\nNote: {note}")
        }
    }

    pub fn submitted_response(&self) -> PromptResponse {
        if self.is_freeform() {
            return PromptResponse::Text {
                text: self.reply_text.clone(),
            };
        }

        let note = self.response_note();
        if self.is_multi() {
            let selections = self
                .selected
                .iter()
                .filter_map(|idx| self.option_label(*idx).map(str::to_string))
                .collect();
            return PromptResponse::Multi { selections, note };
        }

        let selection = self
            .selected_option_idx()
            .and_then(|idx| self.option_label(idx))
            .unwrap_or_default()
            .to_string();
        PromptResponse::Single { selection, note }
    }

    pub fn dismissed_response(&self) -> PromptResponse {
        self.request.empty_response()
    }

    pub fn display_response(&self, response: &PromptResponse) -> String {
        match response {
            PromptResponse::Text { text } => text.clone(),
            PromptResponse::Single { selection, note } => {
                let base = self
                    .request
                    .options
                    .iter()
                    .position(|option| option == selection)
                    .map(|idx| format!("{}. {}", idx + 1, selection))
                    .unwrap_or_else(|| selection.clone());
                Self::format_note_display(base, note.as_deref())
            }
            PromptResponse::Multi { selections, note } => {
                let base = if selections.is_empty() {
                    String::new()
                } else {
                    selections
                        .iter()
                        .filter_map(|selection| {
                            self.request
                                .options
                                .iter()
                                .position(|option| option == selection)
                                .map(|idx| format!("{}. {}", idx + 1, selection))
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                Self::format_note_display(base, note.as_deref())
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct PickerState<T> {
    pub items: Vec<T>,
    pub selected: usize,
}

impl<T> PickerState<T> {
    pub fn new(items: Vec<T>) -> Self {
        Self { items, selected: 0 }
    }

    pub fn up(&mut self) {
        if !self.items.is_empty() {
            self.selected = self.selected.saturating_sub(1);
        }
    }

    pub fn down(&mut self) {
        if !self.items.is_empty() {
            self.selected = (self.selected + 1).min(self.items.len() - 1);
        }
    }

    pub fn take_selected(&mut self) -> Option<T> {
        if self.items.is_empty() {
            return None;
        }
        let idx = self.selected.min(self.items.len() - 1);
        Some(self.items.remove(idx))
    }
}

#[derive(Clone, Debug)]
pub struct TreeRow {
    pub node_id: String,
    pub parent_node_id: Option<String>,
    pub message: Message,
    pub depth: usize,
    pub has_children: bool,
    pub collapsed: bool,
    pub active: bool,
}

#[derive(Clone, Debug)]
pub struct TreeSelection {
    pub node_id: String,
    pub parent_node_id: Option<String>,
    pub message: Message,
}

impl PartialEq for TreeSelection {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id && self.parent_node_id == other.parent_node_id
    }
}

impl Eq for TreeSelection {}

#[derive(Debug)]
pub struct TreeState {
    pub roots: Vec<SessionMessageTreeNode>,
    pub collapsed: BTreeSet<String>,
    pub selected_node_id: Option<String>,
}

impl TreeState {
    pub fn new(roots: Vec<SessionMessageTreeNode>) -> Self {
        // Strip infrastructure nodes before the tree picker sees them.
        // The picker is a conversation navigator — it should only show
        // real user turns and assistant prose responses. Tool-call
        // scaffolding, synthetic tool-result messages, system notices,
        // and compaction summaries are all plumbing and cause broken
        // navigation if a user selects one.
        let roots = filter_user_visible(&roots);
        let selected_node_id = active_node_id(&roots).or_else(|| first_node_id(&roots));
        Self {
            roots,
            collapsed: BTreeSet::new(),
            selected_node_id,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    pub fn rows(&self) -> Vec<TreeRow> {
        let mut rows = Vec::new();
        for node in &self.roots {
            flatten_tree_rows(node, 0, &self.collapsed, &mut rows);
        }
        rows
    }

    pub fn up(&mut self) {
        let rows = self.rows();
        let Some(current_idx) = self.selected_index(&rows) else {
            return;
        };
        let next_idx = current_idx.saturating_sub(1);
        self.selected_node_id = rows.get(next_idx).map(|row| row.node_id.clone());
    }

    pub fn down(&mut self) {
        let rows = self.rows();
        let Some(current_idx) = self.selected_index(&rows) else {
            return;
        };
        let next_idx = (current_idx + 1).min(rows.len().saturating_sub(1));
        self.selected_node_id = rows.get(next_idx).map(|row| row.node_id.clone());
    }

    pub fn collapse_or_jump_prev_branch(&mut self) {
        let rows = self.rows();
        let Some(current_idx) = self.selected_index(&rows) else {
            return;
        };
        let current = &rows[current_idx];
        if current.has_children && !current.collapsed {
            self.collapsed.insert(current.node_id.clone());
            return;
        }
        if let Some(target) = rows[..current_idx]
            .iter()
            .rev()
            .find(|row| row.has_children)
            .map(|row| row.node_id.clone())
        {
            self.selected_node_id = Some(target);
        } else if let Some(parent) = current.parent_node_id.clone() {
            self.selected_node_id = Some(parent);
        }
    }

    pub fn expand_or_jump_next_branch(&mut self) {
        let rows = self.rows();
        let Some(current_idx) = self.selected_index(&rows) else {
            return;
        };
        let current = &rows[current_idx];
        if current.has_children && current.collapsed {
            self.collapsed.remove(&current.node_id);
            return;
        }
        if let Some(target) = rows
            .iter()
            .skip(current_idx + 1)
            .find(|row| row.has_children)
            .map(|row| row.node_id.clone())
        {
            self.selected_node_id = Some(target);
        }
    }

    pub fn take_selected(&mut self) -> Option<TreeSelection> {
        let selected_id = self.selected_node_id.as_deref()?;
        self.rows()
            .into_iter()
            .find(|row| row.node_id == selected_id)
            .map(|row| TreeSelection {
                node_id: row.node_id,
                parent_node_id: row.parent_node_id,
                message: row.message,
            })
    }

    fn selected_index(&self, rows: &[TreeRow]) -> Option<usize> {
        let selected_id = self.selected_node_id.as_deref()?;
        rows.iter().position(|row| row.node_id == selected_id)
    }
}

/// Return the subset of a `SessionMessageTreeNode` list that the tree
/// picker should actually show: real typed user turns and assistant
/// prose responses. Anything else — assistant tool-call scaffolding,
/// synthetic user tool-result messages, system notices, compaction
/// summaries — is infrastructure and gets filtered out.
///
/// Filtered nodes don't hide their descendants: their visible children
/// are spliced up to take the filtered node's place in the parent's
/// children list. This preserves the shape of the visible conversation
/// even when multiple layers of filtered nodes sit between two
/// user-facing turns. `parent_message_node_id` is preserved unchanged
/// so `switch_to_tree_selection` can still walk back to the structural
/// parent of a selected user message.
fn filter_user_visible(nodes: &[SessionMessageTreeNode]) -> Vec<SessionMessageTreeNode> {
    let mut out = Vec::new();
    for node in nodes {
        let visible_children = filter_user_visible(&node.children);
        if is_user_visible_message(&node.message) {
            out.push(SessionMessageTreeNode {
                node_id: node.node_id.clone(),
                parent_message_node_id: node.parent_message_node_id.clone(),
                message: node.message.clone(),
                timestamp: node.timestamp.clone(),
                children: visible_children,
                active: node.active,
            });
        } else {
            // Skip this node; its visible descendants become
            // children of the caller's current level.
            out.extend(visible_children);
        }
    }
    out
}

fn is_user_visible_message(message: &Message) -> bool {
    match message.role {
        MessageRole::User => {
            // A real typed turn always has `user_input` set. Synthetic
            // user messages that exist solely to deliver tool results
            // back to the LLM have no `user_input` and are filtered.
            message.user_input.is_some()
        }
        MessageRole::Assistant => {
            // Keep assistant messages that contain prose the reader
            // would want to see. Pure tool-call scaffolding (no text
            // or image parts) is filtered — its matching tool result
            // gets filtered too, so the flattened view jumps directly
            // from the prior prose turn to the next.
            message.parts.iter().any(|part| {
                matches!(
                    part.kind,
                    PartKind::Text | PartKind::Prose | PartKind::Image
                )
            })
        }
        MessageRole::System => false,
    }
}

fn flatten_tree_rows(
    node: &SessionMessageTreeNode,
    depth: usize,
    collapsed: &BTreeSet<String>,
    rows: &mut Vec<TreeRow>,
) {
    let has_children = !node.children.is_empty();
    let is_collapsed = has_children && collapsed.contains(&node.node_id);
    rows.push(TreeRow {
        node_id: node.node_id.clone(),
        parent_node_id: node.parent_message_node_id.clone(),
        message: node.message.clone(),
        depth,
        has_children,
        collapsed: is_collapsed,
        active: node.active,
    });
    if is_collapsed {
        return;
    }
    // Only bump the visual depth when the node is an actual fork
    // (more than one child). Linear continuations — the common case
    // for an unbranched conversation — stay at the parent's depth so
    // a 60-turn chain doesn't cascade diagonally off the right edge.
    let child_depth = if node.children.len() > 1 {
        depth + 1
    } else {
        depth
    };
    for child in &node.children {
        flatten_tree_rows(child, child_depth, collapsed, rows);
    }
}

fn active_node_id(nodes: &[SessionMessageTreeNode]) -> Option<String> {
    for node in nodes {
        if node.active {
            return Some(node.node_id.clone());
        }
        if let Some(found) = active_node_id(&node.children) {
            return Some(found);
        }
    }
    None
}

fn first_node_id(nodes: &[SessionMessageTreeNode]) -> Option<String> {
    nodes.first().map(|node| node.node_id.clone())
}

pub fn tree_message_preview(message: &Message) -> String {
    if matches!(message.role, MessageRole::User)
        && let Some(text) = message
            .user_input
            .as_ref()
            .map(|input| input.display_text.trim())
            .filter(|text| !text.is_empty())
    {
        return text.replace('\n', " ");
    }

    let mut preview = String::new();
    for part in &message.parts {
        match part.kind {
            PartKind::Text | PartKind::Prose | PartKind::Code | PartKind::Output => {
                if !part.content.trim().is_empty() {
                    if !preview.is_empty() {
                        preview.push(' ');
                    }
                    preview.push_str(part.content.trim());
                }
            }
            PartKind::Image => {
                if !preview.is_empty() {
                    preview.push(' ');
                }
                preview.push_str("[image]");
            }
            PartKind::ToolCall => {
                if !preview.is_empty() {
                    preview.push(' ');
                }
                preview.push_str("[tool call]");
            }
            PartKind::ToolResult => {
                if !preview.is_empty() {
                    preview.push(' ');
                }
                preview.push_str("[tool result]");
            }
            PartKind::Error => {
                if !preview.is_empty() {
                    preview.push(' ');
                }
                preview.push_str("[error]");
            }
            // Reasoning summaries aren't useful as a message preview —
            // the user wants to see the actual reply, not the thinking.
            PartKind::Reasoning => {}
        }
    }
    preview.replace('\n', " ").trim().to_string()
}

#[derive(Debug)]
pub enum OverlayState {
    SessionPicker(PickerState<SessionInfo>),
    Tree(TreeState),
    SkillPicker(PickerState<(String, String)>),
    Prompt(PromptState),
}
