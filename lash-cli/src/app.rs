use lash::AgentEvent;

/// A renderable block in the scrollable history.
pub enum DisplayBlock {
    UserInput(String),
    AssistantText(String),
    CodeBlock {
        code: String,
        expanded: bool,
    },
    ToolCall {
        name: String,
        success: bool,
        duration_ms: u64,
    },
    CodeOutput {
        output: String,
        error: Option<String>,
    },
    Error(String),
    Splash,
}

/// How many visual rows a single line of text takes when wrapped to `width`.
fn wrapped_line_height(line: &str, width: usize) -> usize {
    if width == 0 {
        return 1;
    }
    let len = line.chars().count();
    if len == 0 {
        1
    } else {
        (len + width - 1) / width
    }
}

/// Sum of wrapped visual rows for a multi-line string, with an optional prefix width per line.
fn wrapped_text_height(text: &str, width: usize, prefix_chars: usize) -> usize {
    let effective = width.saturating_sub(prefix_chars);
    let mut h = 0;
    let mut any = false;
    for line in text.lines() {
        h += wrapped_line_height(line, effective);
        any = true;
    }
    if !any {
        h = 1; // empty string → 1 blank line
    }
    h
}

impl DisplayBlock {
    /// Number of visual lines this block takes when rendered at `width` columns.
    pub fn height(&self, code_expanded: bool, width: usize) -> usize {
        match self {
            DisplayBlock::UserInput(s) => {
                // Each line has "/ " prefix (2 chars)
                wrapped_text_height(s, width, 2) + 1 // +1 blank line after
            }
            DisplayBlock::AssistantText(s) => wrapped_text_height(s, width, 0),
            DisplayBlock::CodeBlock { code, expanded } => {
                let show = if code_expanded { *expanded } else { false };
                if show {
                    // "│ " prefix (2 chars) on code lines + header + footer
                    wrapped_text_height(code, width, 2) + 2
                } else {
                    1 // collapsed single line
                }
            }
            DisplayBlock::ToolCall { .. } => 1,
            DisplayBlock::CodeOutput { output, error } => {
                let mut h = 0;
                if code_expanded && !output.is_empty() {
                    // header + "│ " prefixed lines
                    h += 1 + wrapped_text_height(output, width, 2);
                }
                if let Some(err) = error {
                    // header + "│ " prefixed lines
                    h += 1 + wrapped_text_height(err, width, 2);
                }
                if h > 0 {
                    h += 1; // bottom border
                }
                h
            }
            DisplayBlock::Error(msg) => {
                wrapped_line_height(&format!("Error: {}", msg), width)
            }
            DisplayBlock::Splash => 9,
        }
    }
}

pub struct App {
    pub blocks: Vec<DisplayBlock>,
    pub input: String,
    pub cursor_pos: usize,
    pub scroll_offset: usize,
    pub code_expanded: bool,
    pub running: bool,
    pub model: String,
    pub iteration: usize,
    pub input_history: Vec<String>,
    pub input_history_idx: Option<usize>,
    /// Spinner frame counter
    pub tick: usize,
    /// Latest progress message — shown on status bar next to spinner
    pub status_text: Option<String>,
    /// Buffered TextDelta — only flushed as AssistantText on Done (prose-only response).
    /// Discarded when a CodeBlock arrives (it was intermediate thinking with code fences).
    pending_text: String,
}

impl App {
    pub fn new(model: String) -> Self {
        Self {
            blocks: vec![DisplayBlock::Splash],
            input: String::new(),
            cursor_pos: 0,
            scroll_offset: 0,
            code_expanded: false,
            running: false,
            model,
            iteration: 0,
            input_history: Vec::new(),
            input_history_idx: None,
            tick: 0,
            status_text: None,
            pending_text: String::new(),
        }
    }

    /// Get the current input text and reset input state.
    pub fn take_input(&mut self) -> String {
        let text = self.input.clone();
        if !text.is_empty() {
            self.input_history.push(text.clone());
        }
        self.input.clear();
        self.cursor_pos = 0;
        self.input_history_idx = None;
        text
    }

    /// Process an agent event, updating display blocks.
    pub fn handle_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::TextDelta { content } => {
                // Buffer text — we don't know yet if this is the final prose answer
                // or intermediate thinking with code fences. We'll flush on Done
                // or discard when a CodeBlock arrives.
                self.pending_text.push_str(&content);
            }
            AgentEvent::CodeBlock { code } => {
                // Discard buffered text — it was intermediate thinking with code fences
                self.pending_text.clear();
                self.blocks.push(DisplayBlock::CodeBlock {
                    code,
                    expanded: self.code_expanded,
                });
                self.scroll_to_bottom();
            }
            AgentEvent::ToolCall {
                name,
                success,
                duration_ms,
                ..
            } => {
                self.blocks.push(DisplayBlock::ToolCall {
                    name,
                    success,
                    duration_ms,
                });
                self.scroll_to_bottom();
            }
            AgentEvent::CodeOutput { output, error } => {
                // stdout is the model's debug buffer — only show errors to the user.
                // stdout is still stored for CTRL+O expand view.
                if error.is_some() || !output.is_empty() {
                    self.blocks.push(DisplayBlock::CodeOutput { output, error });
                    self.scroll_to_bottom();
                }
            }
            AgentEvent::Message { text, kind } => {
                if kind == "progress" {
                    self.status_text = Some(text);
                }
            }
            AgentEvent::LlmRequest { iteration, .. } => {
                // New iteration starting — discard any buffered text from previous iteration
                self.pending_text.clear();
                self.iteration = iteration + 1;
            }
            AgentEvent::Done => {
                // Flush buffered text as the final prose answer
                let text = std::mem::take(&mut self.pending_text);
                let text = text.trim().to_string();
                if !text.is_empty() {
                    self.blocks.push(DisplayBlock::AssistantText(text));
                    self.scroll_to_bottom();
                }
                self.running = false;
                self.status_text = None;
            }
            AgentEvent::Error { message } => {
                self.blocks.push(DisplayBlock::Error(message));
                self.scroll_to_bottom();
            }
            AgentEvent::LlmResponse { .. } => {}
        }
    }

    /// Toggle code block expansion.
    pub fn toggle_code_expand(&mut self) {
        self.code_expanded = !self.code_expanded;
        for block in &mut self.blocks {
            if let DisplayBlock::CodeBlock { expanded, .. } = block {
                *expanded = self.code_expanded;
            }
        }
    }

    pub fn scroll_up(&mut self, amount: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(amount);
    }

    pub fn scroll_down(&mut self, amount: usize, viewport_height: usize, viewport_width: usize) {
        let total = self.total_content_height(viewport_width);
        let max_scroll = total.saturating_sub(viewport_height);
        self.scroll_offset = self.scroll_offset.saturating_add(amount).min(max_scroll);
    }

    pub fn scroll_to_bottom(&mut self) {
        // We'll clamp this in rendering when we know viewport height
        self.scroll_offset = usize::MAX;
    }

    pub fn total_content_height(&self, width: usize) -> usize {
        self.blocks
            .iter()
            .map(|b| b.height(self.code_expanded, width))
            .sum()
    }

    /// Navigate input history with up arrow.
    pub fn history_up(&mut self) {
        if self.input_history.is_empty() {
            return;
        }
        let idx = match self.input_history_idx {
            None => self.input_history.len() - 1,
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.input_history_idx = Some(idx);
        self.input = self.input_history[idx].clone();
        self.cursor_pos = self.input.len();
    }

    /// Navigate input history with down arrow.
    pub fn history_down(&mut self) {
        match self.input_history_idx {
            None => {}
            Some(i) if i + 1 >= self.input_history.len() => {
                self.input_history_idx = None;
                self.input.clear();
                self.cursor_pos = 0;
            }
            Some(i) => {
                let idx = i + 1;
                self.input_history_idx = Some(idx);
                self.input = self.input_history[idx].clone();
                self.cursor_pos = self.input.len();
            }
        }
    }

    /// Insert a character at cursor position.
    pub fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor_pos, c);
        self.cursor_pos += c.len_utf8();
    }

    /// Delete character before cursor.
    pub fn backspace(&mut self) {
        if self.cursor_pos > 0 {
            let prev = self.input[..self.cursor_pos]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.input.drain(prev..self.cursor_pos);
            self.cursor_pos = prev;
        }
    }

    /// Delete character at cursor.
    pub fn delete(&mut self) {
        if self.cursor_pos < self.input.len() {
            let next = self.input[self.cursor_pos..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| self.cursor_pos + i)
                .unwrap_or(self.input.len());
            self.input.drain(self.cursor_pos..next);
        }
    }

    pub fn move_cursor_left(&mut self) {
        if self.cursor_pos > 0 {
            self.cursor_pos = self.input[..self.cursor_pos]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
    }

    pub fn move_cursor_right(&mut self) {
        if self.cursor_pos < self.input.len() {
            self.cursor_pos = self.input[self.cursor_pos..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| self.cursor_pos + i)
                .unwrap_or(self.input.len());
        }
    }

    pub fn move_cursor_home(&mut self) {
        self.cursor_pos = 0;
    }

    pub fn move_cursor_end(&mut self) {
        self.cursor_pos = self.input.len();
    }
}
