use std::path::PathBuf;
use std::sync::Arc;

use lash::{AgentEvent, Store, TokenUsage};

use crate::command;
use crate::markdown;
use crate::skill::SkillRegistry;

#[derive(Clone, Debug)]
pub struct TaskSnapshot {
    pub id: String,
    pub label: String,
    pub status: String,
    pub owner: String,
    pub is_blocked: bool,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Normal,
    Plan,
}

/// State for an active agent prompt dialog.
pub struct PromptState {
    pub question: String,
    /// Original options (may be empty for freeform-only prompts).
    pub options: Vec<String>,
    /// 0..options.len() = option, options.len() = "Other"
    pub selected_idx: usize,
    /// Freeform context text (Shift+Tab to activate).
    pub extra_text: String,
    pub extra_cursor: usize,
    /// True when Shift+Tab has been pressed to edit extra text.
    pub editing_extra: bool,
    pub response_tx: std::sync::mpsc::Sender<String>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum SuggestionKind {
    None,
    Command, // slash commands
    Path,    // @ file/dir references
}

/// A renderable block in the scrollable history.
#[allow(dead_code)]
pub enum DisplayBlock {
    UserInput(String),
    AssistantText(String),
    CodeBlock {
        code: String,
        expanded: bool,
        /// If true, this block is a continuation of a prior code-block group.
        /// In collapsed mode it is hidden; the group leader shows total lines.
        continuation: bool,
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
    ShellOutput {
        command: String,
        output: String,
        error: Option<String>,
    },
    Error(String),
    /// Informational message from the system (e.g. /help output).
    SystemMessage(String),
    /// Rendered plan content from plan mode (bordered markdown).
    PlanContent(String),
    /// Sub-agent completion result with token stats.
    SubAgentResult {
        task: String,
        usage: TokenUsage,
        tool_calls: usize,
        iterations: usize,
        success: bool,
        /// Whether this is the last in a consecutive sequence (uses └─ instead of ├─).
        is_last: bool,
    },
    /// Live task list (bordered inline display).
    TaskList {
        tasks: Vec<TaskSnapshot>,
    },
    Splash,
}

/// How many visual rows a single line of text takes when wrapped to `width`.
fn wrapped_line_height(line: &str, width: usize) -> usize {
    if width == 0 {
        return 1;
    }
    let len = unicode_width::UnicodeWidthStr::width(line);
    if len == 0 { 1 } else { len.div_ceil(width) }
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
    /// `viewport_height` is needed for Splash centering; pass 0 for non-Splash blocks.
    pub fn height(&self, code_expanded: bool, width: usize, viewport_height: usize) -> usize {
        match self {
            DisplayBlock::UserInput(s) => {
                // Each line has "/ " prefix (2 chars)
                wrapped_text_height(s, width, 2) + 1 // +1 blank line after
            }
            DisplayBlock::AssistantText(s) => markdown::markdown_height(s, width),
            DisplayBlock::CodeBlock {
                code,
                expanded,
                continuation,
            } => {
                let show = if code_expanded { *expanded } else { false };
                if show {
                    // "│ " prefix (2 chars) on code lines
                    wrapped_text_height(code, width, 2)
                } else if *continuation {
                    0 // hidden in collapsed mode — leader shows total
                } else {
                    1 // collapsed single line
                }
            }
            DisplayBlock::ToolCall { .. } => 1,
            DisplayBlock::CodeOutput { output, error } => {
                let mut h = 0;
                if code_expanded && !output.is_empty() {
                    h += wrapped_text_height(output, width, 2);
                }
                if let Some(err) = error {
                    h += wrapped_text_height(err, width, 2);
                }
                h
            }
            DisplayBlock::ShellOutput { output, error, .. } => {
                let mut h = 1; // "$ command" header
                if !output.is_empty() {
                    h += wrapped_text_height(output, width, 2); // "│ " prefix
                }
                if let Some(err) = error {
                    h += wrapped_text_height(err, width, 2);
                }
                h += 1; // bottom border
                h
            }
            DisplayBlock::Error(msg) => wrapped_line_height(&format!("Error: {}", msg), width),
            DisplayBlock::SystemMessage(s) => wrapped_text_height(s, width, 0) + 1, // +1 blank after
            DisplayBlock::SubAgentResult { .. } => 2, // task line + status line
            DisplayBlock::PlanContent(s) => {
                // borders (2) + title line is part of top border + markdown content height
                2 + markdown::markdown_height(s, width.saturating_sub(2))
            }
            DisplayBlock::TaskList { tasks } => {
                2 + tasks.len() // top border + tasks + bottom border
            }
            DisplayBlock::Splash => {
                // Splash fills the entire viewport (centered content + top/bottom padding)
                viewport_height
            }
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
    /// Buffered TextDelta — rendered live as streaming text, flushed as AssistantText on Done.
    /// Discarded when a CodeBlock arrives (it was intermediate thinking with code fences).
    pub pending_text: String,
    /// Active suggestions (display text, description).
    pub suggestions: Vec<(String, String)>,
    /// Currently selected suggestion index.
    pub suggestion_idx: usize,
    /// What kind of suggestion is currently active.
    pub suggestion_kind: SuggestionKind,
    /// Session picker: list of session info when browsing sessions.
    pub session_picker: Vec<crate::session_log::SessionInfo>,
    /// Currently selected session index.
    pub session_picker_idx: usize,
    /// Whether the UI needs a redraw.
    pub dirty: bool,
    /// Auto-scroll to bottom on new output. Disabled when user scrolls up during generation.
    pub follow_output: bool,
    /// Cumulative height prefix sums for each block (used for O(1) total height and O(log n) lookup).
    height_cache: Vec<usize>,
    /// Terminal width the height cache was computed for.
    height_cache_width: usize,
    /// Viewport height the height cache was computed for (needed for Splash centering).
    height_cache_vh: usize,
    /// PNG-encoded images pasted via Ctrl+V, waiting to be sent with next message.
    pub pending_images: Vec<Vec<u8>>,
    /// Live streaming output lines from tool execution (e.g. bash).
    pub streaming_output: Vec<String>,
    /// Loaded skills registry.
    pub skills: SkillRegistry,
    /// Skill picker: list of (name, description) when browsing skills.
    pub skill_picker: Vec<(String, String)>,
    /// Currently selected skill index.
    pub skill_picker_idx: usize,
    /// Messages queued while agent is running, sent sequentially on completion.
    pub message_queue: Vec<String>,
    /// Active agent prompt (ask() dialog).
    pub prompt: Option<PromptState>,
    /// Whether the terminal window is currently focused.
    pub focused: bool,
    /// Cumulative token usage for the current session.
    pub token_usage: TokenUsage,
    /// Context window size for the current model (from models.dev).
    pub context_window: Option<u64>,
    /// Latest iteration's input tokens (current context size, not cumulative).
    pub last_input_tokens: i64,
    /// Current mode: Normal (read-write) or Plan (explore-only).
    pub mode: Mode,
    /// Path to the plan file for this session (generated on first plan entry).
    pub plan_file: Option<PathBuf>,
    /// Last modification time of the plan file, for change detection.
    pub plan_file_mtime: Option<std::time::SystemTime>,
    /// Unique session name (e.g. "alpine-canyon").
    pub session_name: String,
    /// Current working directory with ~ substitution.
    pub cwd: String,
    /// Store handle for reading task data (shared with agent).
    pub store: Option<Arc<Store>>,
    /// Snapshots for the persistent task tray (bottom panel).
    pub task_tray: Vec<TaskSnapshot>,
    /// Active delegate sub-agent: (name, task description, started_at).
    pub active_delegate: Option<(String, String, std::time::Instant)>,
}

impl App {
    pub fn new(model: String, session_name: String, store: Option<Arc<Store>>) -> Self {
        let context_window = lash::model_info::context_window(&model);
        let cwd = {
            let home = std::env::var("HOME").unwrap_or_default();
            let dir = std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| ".".into());
            if !home.is_empty() && dir.starts_with(&home) {
                format!("~{}", &dir[home.len()..])
            } else {
                dir
            }
        };
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
            suggestions: Vec::new(),
            suggestion_idx: 0,
            suggestion_kind: SuggestionKind::None,
            session_picker: Vec::new(),
            session_picker_idx: 0,
            dirty: true,
            follow_output: true,
            height_cache: Vec::new(),
            height_cache_width: 0,
            height_cache_vh: 0,
            pending_images: Vec::new(),
            streaming_output: Vec::new(),
            skills: SkillRegistry::load(),
            skill_picker: Vec::new(),
            skill_picker_idx: 0,
            message_queue: Vec::new(),
            prompt: None,
            focused: true,
            token_usage: TokenUsage::default(),
            context_window,
            last_input_tokens: 0,
            mode: Mode::Normal,
            plan_file: None,
            plan_file_mtime: None,
            session_name,
            cwd,
            store,
            task_tray: Vec::new(),
            active_delegate: None,
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
        self.follow_output = true;
        text
    }

    /// Take pending images, clearing them from the app.
    pub fn take_images(&mut self) -> Vec<Vec<u8>> {
        std::mem::take(&mut self.pending_images)
    }

    /// Mark the height cache as stale so it will be recomputed on next access.
    pub fn invalidate_height_cache(&mut self) {
        self.height_cache.clear();
    }

    /// Check whether a new CodeBlock belongs to an existing code-block group.
    /// Returns `true` if there is a prior CodeBlock with only ToolCall /
    /// CodeOutput / TaskList blocks between it and the end (no user-facing
    /// boundary like AssistantText, Error, UserInput, etc.).
    fn is_code_continuation(&self) -> bool {
        for block in self.blocks.iter().rev() {
            match block {
                DisplayBlock::CodeBlock { .. } => return true,
                DisplayBlock::ToolCall { .. }
                | DisplayBlock::CodeOutput { .. }
                | DisplayBlock::TaskList { .. } => continue,
                _ => return false,
            }
        }
        false
    }

    /// Process an agent event, updating display blocks.
    pub fn handle_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::TextDelta { content } => {
                self.pending_text.push_str(&content);
                self.scroll_to_bottom();
            }
            AgentEvent::CodeBlock { code } => {
                self.pending_text.clear();
                let trimmed = code.trim_matches('\n');
                if !trimmed.is_empty() {
                    let continuation = self.is_code_continuation();
                    self.blocks.push(DisplayBlock::CodeBlock {
                        code: trimmed.to_string(),
                        expanded: self.code_expanded,
                        continuation,
                    });
                    self.invalidate_height_cache();
                }
                self.scroll_to_bottom();
            }
            AgentEvent::ToolCall {
                name,
                success,
                duration_ms,
                ..
            } => {
                self.streaming_output.clear();
                if name.starts_with("delegate_") {
                    self.active_delegate = None;
                }
                let is_task_tool = matches!(
                    name.as_str(),
                    "create_task" | "update_task" | "claim_task" | "delete_task"
                );
                self.blocks.push(DisplayBlock::ToolCall {
                    name,
                    success,
                    duration_ms,
                });
                self.invalidate_height_cache();
                if is_task_tool {
                    self.refresh_tasks();
                }
                self.scroll_to_bottom();
            }
            AgentEvent::CodeOutput { output, error } => {
                if error.is_some() || !output.is_empty() {
                    self.blocks.push(DisplayBlock::CodeOutput { output, error });
                    self.invalidate_height_cache();
                }
                self.scroll_to_bottom();
            }
            AgentEvent::Message { text, kind } => {
                if kind == "delegate_start" {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                        let name = v
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("delegate")
                            .to_string();
                        let task = v
                            .get("task")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        self.active_delegate = Some((name, task, std::time::Instant::now()));
                    }
                    self.scroll_to_bottom();
                } else if kind == "tool_output" {
                    self.streaming_output.push(text);
                    self.scroll_to_bottom();
                } else {
                    self.blocks.push(DisplayBlock::AssistantText(text));
                    self.invalidate_height_cache();
                    self.scroll_to_bottom();
                }
            }
            AgentEvent::LlmRequest { iteration, .. } => {
                self.pending_text.clear();
                self.iteration = iteration + 1;
            }
            AgentEvent::Done => {
                self.pending_text.clear();
                self.running = false;
                self.status_text = None;
                self.streaming_output.clear();
                self.active_delegate = None;
            }
            AgentEvent::Error { message } => {
                self.blocks.push(DisplayBlock::Error(message));
                self.invalidate_height_cache();
                self.scroll_to_bottom();
            }
            AgentEvent::TokenUsage {
                usage, cumulative, ..
            } => {
                self.token_usage = cumulative;
                self.last_input_tokens = usage.input_tokens;
            }
            AgentEvent::SubAgentDone {
                task,
                usage,
                tool_calls,
                iterations,
                success,
            } => {
                // Mark previous SubAgentResult blocks as not-last, then add new one as last
                for block in self.blocks.iter_mut().rev() {
                    match block {
                        DisplayBlock::SubAgentResult { is_last, .. } if *is_last => {
                            *is_last = false;
                        }
                        DisplayBlock::SubAgentResult { .. } => break,
                        _ => break,
                    }
                }
                self.blocks.push(DisplayBlock::SubAgentResult {
                    task,
                    usage,
                    tool_calls,
                    iterations,
                    success,
                    is_last: true,
                });
                self.invalidate_height_cache();
                self.scroll_to_bottom();
            }
            AgentEvent::LlmResponse { .. } => {}
            AgentEvent::Prompt { .. } => {
                // Handled by the main event loop, not here
            }
        }
    }

    pub fn queue_message(&mut self, msg: String) {
        self.message_queue.push(msg);
    }

    pub fn dequeue_message(&mut self) -> Option<String> {
        if self.message_queue.is_empty() {
            None
        } else {
            Some(self.message_queue.remove(0))
        }
    }

    /// Pop last queued message back to editor for editing/removal.
    pub fn unqueue_last(&mut self) -> Option<String> {
        self.message_queue.pop()
    }

    pub fn queue_count(&self) -> usize {
        self.message_queue.len()
    }

    /// Reset conversation to initial splash screen.
    pub fn clear(&mut self) {
        self.blocks = vec![DisplayBlock::Splash];
        self.scroll_offset = 0;
        self.follow_output = true;
        self.pending_text.clear();
        self.status_text = None;
        self.pending_images.clear();
        self.streaming_output.clear();
        self.message_queue.clear();
        self.task_tray.clear();
        self.active_delegate = None;
        self.token_usage = TokenUsage::default();
        self.last_input_tokens = 0;
        self.invalidate_height_cache();
    }

    /// Ensure a plan file path exists, creating the directory and returning the path.
    /// Generated on first call per session; subsequent calls return the same path.
    pub fn ensure_plan_file(&mut self) -> PathBuf {
        if let Some(ref path) = self.plan_file {
            return path.clone();
        }

        // Detect git repo root by walking up from cwd
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let base = find_git_root(&cwd)
            .map(|root| root.join(".lash"))
            .unwrap_or_else(|| {
                let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
                PathBuf::from(home).join(".lash")
            });
        let plans_dir = base.join("plans");
        let _ = std::fs::create_dir_all(&plans_dir);

        // Generate slug: YYYYMMDD_HHMMSS-adjective-noun
        let now = chrono::Local::now();
        let timestamp = now.format("%Y%m%d_%H%M%S");
        let slug = generate_plan_slug();
        let filename = format!("{}-{}.md", timestamp, slug);
        let path = plans_dir.join(filename);

        self.plan_file = Some(path.clone());
        path
    }

    /// Toggle code block expansion with scroll anchoring.
    ///
    /// When the user is scrolled to a specific position, we anchor to the block
    /// at the top of the viewport so the same content stays visible after
    /// code blocks change height.
    pub fn toggle_code_expand(&mut self) {
        // Determine anchor before toggling
        let anchor = if self.follow_output || self.height_cache_width == 0 {
            None
        } else if !self.height_cache.is_empty() {
            let cache = &self.height_cache;
            let idx = cache.partition_point(|&cum| cum <= self.scroll_offset);
            if idx >= self.blocks.len() {
                None
            } else {
                let block_start = if idx == 0 { 0 } else { cache[idx - 1] };
                let skip = self.scroll_offset - block_start;
                Some((idx, skip))
            }
        } else {
            None
        };

        // Toggle expansion state
        self.code_expanded = !self.code_expanded;
        for block in &mut self.blocks {
            if let DisplayBlock::CodeBlock { expanded, .. } = block {
                *expanded = self.code_expanded;
            }
        }
        self.invalidate_height_cache();

        // Restore scroll to keep the anchor block visible
        if let Some((anchor_idx, anchor_skip)) = anchor {
            let w = self.height_cache_width;
            let vh = self.height_cache_vh;
            self.ensure_height_cache(w, vh);

            let new_block_start = if anchor_idx == 0 {
                0
            } else {
                self.height_cache[anchor_idx - 1]
            };
            let new_block_height = self.height_cache[anchor_idx] - new_block_start;
            let clamped_skip = anchor_skip.min(new_block_height.saturating_sub(1));
            self.scroll_offset = new_block_start + clamped_skip;
        }
    }

    /// Refresh the task list display block from the Store.
    pub fn refresh_tasks(&mut self) {
        let store = match &self.store {
            Some(s) => s,
            None => return,
        };

        let all_tasks = store.list_tasks(None, None);
        if all_tasks.is_empty() {
            // Remove existing TaskList block if any
            self.blocks
                .retain(|b| !matches!(b, DisplayBlock::TaskList { .. }));
            self.task_tray.clear();
            self.invalidate_height_cache();
            return;
        }

        let mut snapshots: Vec<TaskSnapshot> = all_tasks
            .iter()
            .map(|t| {
                let is_blocked = !t.blocked_by.is_empty()
                    && t.blocked_by.iter().any(|bid| {
                        all_tasks.iter().any(|bt| {
                            bt.id == *bid && bt.status != "completed" && bt.status != "cancelled"
                        })
                    });
                let label = if t.status == "in_progress" && !t.active_form.is_empty() {
                    t.active_form.clone()
                } else {
                    t.subject.clone()
                };
                TaskSnapshot {
                    id: t.id.clone(),
                    label,
                    status: t.status.clone(),
                    owner: t.owner.clone(),
                    is_blocked,
                }
            })
            .collect();

        // Sort: in_progress first, then pending, then completed/cancelled. Within group, by ID.
        snapshots.sort_by(|a, b| {
            fn status_order(s: &str) -> u8 {
                match s {
                    "in_progress" => 0,
                    "pending" => 1,
                    "completed" => 2,
                    "cancelled" => 3,
                    _ => 4,
                }
            }
            status_order(&a.status)
                .cmp(&status_order(&b.status))
                .then_with(|| a.id.cmp(&b.id))
        });

        self.task_tray = snapshots.clone();

        let new_block = DisplayBlock::TaskList { tasks: snapshots };

        // Replace existing TaskList in-place, or push new one
        let existing_idx = self
            .blocks
            .iter()
            .rposition(|b| matches!(b, DisplayBlock::TaskList { .. }));
        if let Some(idx) = existing_idx {
            self.blocks[idx] = new_block;
        } else {
            self.blocks.push(new_block);
        }

        self.invalidate_height_cache();
    }

    /// Height of the persistent task tray: 0/3/4 lines.
    pub fn task_tray_height(&self) -> u16 {
        if self.task_tray.is_empty() {
            0
        } else if self.task_tray.len() == 1 {
            3 // top border + pills + bottom border
        } else {
            4 // top border + pills + progress bar + bottom border
        }
    }

    pub fn scroll_up(&mut self, amount: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(amount);
        self.follow_output = false;
    }

    pub fn scroll_down(&mut self, amount: usize, viewport_height: usize, viewport_width: usize) {
        let total = self.total_content_height(viewport_width, viewport_height);
        let max_scroll = total.saturating_sub(viewport_height);
        self.scroll_offset = self.scroll_offset.saturating_add(amount).min(max_scroll);
        // Re-enable follow if we've reached the bottom
        if self.scroll_offset >= max_scroll {
            self.follow_output = true;
        }
    }

    pub fn scroll_to_bottom(&mut self) {
        if !self.follow_output {
            return;
        }
        // We'll clamp this in rendering when we know viewport height
        self.scroll_offset = usize::MAX;
    }

    /// Public accessor to pre-warm the height cache before an immutable borrow (e.g. draw).
    pub fn ensure_height_cache_pub(&mut self, width: usize, viewport_height: usize) {
        self.ensure_height_cache(width, viewport_height);
    }

    /// Read-only view of the height cache (must be pre-warmed).
    pub fn height_cache_snapshot(&self) -> &[usize] {
        &self.height_cache
    }

    /// Ensure the height cache prefix sums are up to date for the given dimensions.
    fn ensure_height_cache(&mut self, width: usize, viewport_height: usize) {
        if !self.height_cache.is_empty()
            && self.height_cache_width == width
            && self.height_cache_vh == viewport_height
        {
            return;
        }
        self.height_cache_width = width;
        self.height_cache_vh = viewport_height;
        self.height_cache.clear();
        self.height_cache.reserve(self.blocks.len());
        let mut cumulative: usize = 0;
        for block in &self.blocks {
            cumulative += block.height(self.code_expanded, width, viewport_height);
            self.height_cache.push(cumulative);
        }
    }

    pub fn total_content_height(&mut self, width: usize, viewport_height: usize) -> usize {
        self.ensure_height_cache(width, viewport_height);
        let block_height = self.height_cache.last().copied().unwrap_or(0);
        // Include live streaming output lines
        let streaming_height = self.streaming_output.len();
        // Include active delegate indicator line
        let delegate_height = if self.active_delegate.is_some() { 1 } else { 0 };
        block_height + streaming_height + delegate_height
    }

    /// Which line (0-indexed) the cursor is on in multi-line input.
    pub fn cursor_line(&self) -> usize {
        self.input[..self.cursor_pos].matches('\n').count()
    }

    /// Total number of lines in the input.
    pub fn line_count(&self) -> usize {
        self.input.split('\n').count()
    }

    /// Navigate input history with up arrow.
    /// In multi-line mode, if cursor is not on the first line, moves cursor up instead.
    pub fn history_up(&mut self) {
        // If multi-line and not on first line, move cursor up
        if self.cursor_line() > 0 {
            self.move_cursor_up_line();
            return;
        }
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
    /// In multi-line mode, if cursor is not on the last line, moves cursor down instead.
    pub fn history_down(&mut self) {
        // If multi-line and not on last line, move cursor down
        if self.cursor_line() < self.line_count() - 1 {
            self.move_cursor_down_line();
            return;
        }
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

    /// Move cursor up one line within multi-line input.
    fn move_cursor_up_line(&mut self) {
        let before = &self.input[..self.cursor_pos];
        // Find the start of current line
        let cur_line_start = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
        let col = self.cursor_pos - cur_line_start;
        // Find the start of the previous line
        if cur_line_start == 0 {
            return;
        }
        let prev_content = &self.input[..cur_line_start - 1]; // before the \n
        let prev_line_start = prev_content.rfind('\n').map(|i| i + 1).unwrap_or(0);
        let prev_line_len = cur_line_start - 1 - prev_line_start;
        self.cursor_pos = prev_line_start + col.min(prev_line_len);
    }

    /// Move cursor down one line within multi-line input.
    fn move_cursor_down_line(&mut self) {
        let before = &self.input[..self.cursor_pos];
        let cur_line_start = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
        let col = self.cursor_pos - cur_line_start;
        // Find the end of current line (position of \n)
        let after = &self.input[self.cursor_pos..];
        let newline_offset = match after.find('\n') {
            Some(i) => i,
            None => return,
        };
        let next_line_start = self.cursor_pos + newline_offset + 1;
        let next_after = &self.input[next_line_start..];
        let next_line_len = next_after.find('\n').unwrap_or(next_after.len());
        self.cursor_pos = next_line_start + col.min(next_line_len);
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
        // Go to start of current line (not start of entire input)
        let before = &self.input[..self.cursor_pos];
        self.cursor_pos = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
    }

    pub fn move_cursor_end(&mut self) {
        // Go to end of current line (not end of entire input)
        let after = &self.input[self.cursor_pos..];
        if let Some(pos) = after.find('\n') {
            self.cursor_pos += pos;
        } else {
            self.cursor_pos = self.input.len();
        }
    }

    /// Load input history from ~/.lash/history.
    pub fn load_history(&mut self) {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let path = PathBuf::from(home).join(".lash").join("history");
        if let Ok(content) = std::fs::read_to_string(&path) {
            self.input_history = content
                .lines()
                .filter(|l| !l.is_empty())
                .map(|l| l.to_string())
                .collect();
        }
    }

    /// Save input history to ~/.lash/history (last 500 entries).
    pub fn save_history(&self) {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let dir = PathBuf::from(home).join(".lash");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("history");
        let start = self.input_history.len().saturating_sub(500);
        let lines: Vec<&str> = self.input_history[start..]
            .iter()
            .map(|s| s.as_str())
            .collect();
        let _ = std::fs::write(&path, lines.join("\n"));
    }

    /// Update the suggestion list based on current input.
    pub fn update_suggestions(&mut self) {
        // 1. Check slash commands first
        if self.input.starts_with('/') && !self.input.contains(' ') {
            self.suggestions = command::completions(&self.input, &self.skills);
            self.suggestion_kind = SuggestionKind::Command;
            if self.suggestions.is_empty() {
                self.suggestion_idx = 0;
            } else {
                self.suggestion_idx = self.suggestion_idx.min(self.suggestions.len() - 1);
            }
            return;
        }
        // 2. Check for @ path token at cursor
        if let Some((_at_pos, partial)) = self.at_token_at_cursor() {
            self.suggestions = complete_path(&partial);
            self.suggestion_kind = SuggestionKind::Path;
            if self.suggestions.is_empty() {
                self.suggestion_idx = 0;
            } else {
                self.suggestion_idx = self.suggestion_idx.min(self.suggestions.len() - 1);
            }
            return;
        }
        // 3. No suggestions
        self.suggestions.clear();
        self.suggestion_idx = 0;
        self.suggestion_kind = SuggestionKind::None;
    }

    /// Scan backwards from cursor to find the nearest `@` token.
    /// Returns `Some((at_byte_pos, partial_path_str))` or `None`.
    /// The `@` must be at start of input or preceded by whitespace.
    /// The partial is everything from after `@` to cursor (spaces end the token).
    fn at_token_at_cursor(&self) -> Option<(usize, String)> {
        let before = &self.input[..self.cursor_pos];
        // Scan backwards for '@'
        let at_byte = before.rfind('@')?;
        // '@' must be at start or preceded by whitespace
        if at_byte > 0 {
            let prev_byte = self.input.as_bytes()[at_byte - 1];
            if !prev_byte.is_ascii_whitespace() {
                return None;
            }
        }
        // Partial is everything after '@' to cursor — must have no spaces
        let partial = &self.input[at_byte + 1..self.cursor_pos];
        if partial.contains(' ') || partial.contains('\n') {
            return None;
        }
        Some((at_byte, partial.to_string()))
    }

    /// Whether the suggestion popup is active.
    pub fn has_suggestions(&self) -> bool {
        !self.suggestions.is_empty()
    }

    /// Move suggestion selection up.
    pub fn suggestion_up(&mut self) {
        if !self.suggestions.is_empty() {
            self.suggestion_idx = self.suggestion_idx.saturating_sub(1);
        }
    }

    /// Move suggestion selection down.
    pub fn suggestion_down(&mut self) {
        if !self.suggestions.is_empty() {
            self.suggestion_idx = (self.suggestion_idx + 1).min(self.suggestions.len() - 1);
        }
    }

    /// Accept the selected suggestion.
    pub fn complete_suggestion(&mut self) {
        match self.suggestion_kind {
            SuggestionKind::Command => {
                if let Some((cmd, _)) = self.suggestions.get(self.suggestion_idx).cloned() {
                    let needs_arg =
                        cmd == "/model" || self.skills.get(cmd.trim_start_matches('/')).is_some();
                    self.input = if needs_arg { format!("{} ", cmd) } else { cmd };
                    self.cursor_pos = self.input.len();
                }
                self.suggestions.clear();
                self.suggestion_idx = 0;
                self.suggestion_kind = SuggestionKind::None;
            }
            SuggestionKind::Path => {
                if let Some((at_pos, _partial)) = self.at_token_at_cursor()
                    && let Some((path, _)) = self.suggestions.get(self.suggestion_idx).cloned()
                {
                    let before = self.input[..at_pos].to_string();
                    let after = self.input[self.cursor_pos..].to_string();
                    let is_dir = path.ends_with('/');
                    self.input = format!("{}@{}{}", before, path, after);
                    self.cursor_pos = at_pos + 1 + path.len(); // +1 for @
                    if is_dir {
                        // Don't dismiss — re-trigger for deeper completion
                        return;
                    }
                }
                self.suggestions.clear();
                self.suggestion_idx = 0;
                self.suggestion_kind = SuggestionKind::None;
            }
            SuggestionKind::None => {}
        }
    }

    /// Whether the session picker is active.
    pub fn has_session_picker(&self) -> bool {
        !self.session_picker.is_empty()
    }

    /// Move session picker selection up.
    pub fn session_picker_up(&mut self) {
        if !self.session_picker.is_empty() {
            self.session_picker_idx = self.session_picker_idx.saturating_sub(1);
        }
    }

    /// Move session picker selection down.
    pub fn session_picker_down(&mut self) {
        if !self.session_picker.is_empty() {
            self.session_picker_idx =
                (self.session_picker_idx + 1).min(self.session_picker.len() - 1);
        }
    }

    /// Get the selected session filename, clearing the picker.
    pub fn take_session_pick(&mut self) -> Option<String> {
        let filename = self
            .session_picker
            .get(self.session_picker_idx)
            .map(|s| s.filename.clone());
        self.session_picker.clear();
        self.session_picker_idx = 0;
        filename
    }

    /// Dismiss the session picker without selecting.
    pub fn dismiss_session_picker(&mut self) {
        self.session_picker.clear();
        self.session_picker_idx = 0;
    }

    /// Whether the skill picker is active.
    pub fn has_skill_picker(&self) -> bool {
        !self.skill_picker.is_empty()
    }

    /// Move skill picker selection up.
    pub fn skill_picker_up(&mut self) {
        if !self.skill_picker.is_empty() {
            self.skill_picker_idx = self.skill_picker_idx.saturating_sub(1);
        }
    }

    /// Move skill picker selection down.
    pub fn skill_picker_down(&mut self) {
        if !self.skill_picker.is_empty() {
            self.skill_picker_idx = (self.skill_picker_idx + 1).min(self.skill_picker.len() - 1);
        }
    }

    /// Get the selected skill name, clearing the picker.
    pub fn take_skill_pick(&mut self) -> Option<String> {
        let name = self
            .skill_picker
            .get(self.skill_picker_idx)
            .map(|(n, _)| n.clone());
        self.skill_picker.clear();
        self.skill_picker_idx = 0;
        name
    }

    /// Dismiss the skill picker without selecting.
    pub fn dismiss_skill_picker(&mut self) {
        self.skill_picker.clear();
        self.skill_picker_idx = 0;
    }

    // ── Prompt (ask dialog) methods ──

    /// Whether the prompt dialog is active.
    pub fn has_prompt(&self) -> bool {
        self.prompt.is_some()
    }

    /// Whether the prompt is in extra-text editing mode.
    pub fn is_prompt_editing_extra(&self) -> bool {
        self.prompt.as_ref().is_some_and(|p| p.editing_extra)
    }

    /// Whether the prompt is freeform-only (no options).
    pub fn is_prompt_freeform(&self) -> bool {
        self.prompt.as_ref().is_some_and(|p| p.options.is_empty())
    }

    /// Move prompt selection up.
    pub fn prompt_up(&mut self) {
        if let Some(p) = &mut self.prompt
            && !p.options.is_empty()
        {
            p.selected_idx = p.selected_idx.saturating_sub(1);
        }
    }

    /// Move prompt selection down.
    pub fn prompt_down(&mut self) {
        if let Some(p) = &mut self.prompt
            && !p.options.is_empty()
        {
            // options.len() = "Other" index
            p.selected_idx = (p.selected_idx + 1).min(p.options.len());
        }
    }

    /// Toggle extra text editing (Shift+Tab).
    pub fn prompt_toggle_extra(&mut self) {
        if let Some(p) = &mut self.prompt {
            p.editing_extra = !p.editing_extra;
        }
    }

    /// Insert a character into the prompt extra text (or freeform input).
    pub fn prompt_insert_char(&mut self, c: char) {
        if let Some(p) = &mut self.prompt {
            p.extra_text.insert(p.extra_cursor, c);
            p.extra_cursor += c.len_utf8();
        }
    }

    /// Delete character before cursor in prompt extra text.
    pub fn prompt_backspace(&mut self) {
        if let Some(p) = &mut self.prompt
            && p.extra_cursor > 0
        {
            let prev = p.extra_text[..p.extra_cursor]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
            p.extra_text.drain(prev..p.extra_cursor);
            p.extra_cursor = prev;
        }
    }

    /// Submit the prompt response and dismiss the dialog.
    pub fn take_prompt_response(&mut self) {
        if let Some(p) = self.prompt.take() {
            let response = if p.options.is_empty() {
                // Freeform-only: just the text
                p.extra_text
            } else if p.selected_idx < p.options.len() {
                // Option selected
                let label = &p.options[p.selected_idx];
                let truncated: String = label.chars().take(40).collect();
                let suffix = if label.chars().count() > 40 {
                    "..."
                } else {
                    ""
                };
                let base = format!("{}. {}{}", p.selected_idx + 1, truncated, suffix);
                if p.extra_text.is_empty() {
                    base
                } else {
                    format!("{}\n\n{}", base, p.extra_text)
                }
            } else {
                // "Other" selected
                p.extra_text
            };
            let _ = p.response_tx.send(response);
        }
    }

    /// Dismiss the prompt without selecting (Esc) — sends empty string to unblock Python.
    pub fn dismiss_prompt(&mut self) {
        if let Some(p) = self.prompt.take() {
            let _ = p.response_tx.send(String::new());
        }
    }
}

/// Complete a partial path for `@` references.
/// Returns up to 20 entries as `(display_name, kind_label)`.
/// Directories get a trailing `/` and are sorted first.
fn complete_path(partial: &str) -> Vec<(String, String)> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let (dir, prefix) = if partial.is_empty() {
        (cwd.clone(), String::new())
    } else if partial.ends_with('/') {
        // e.g. "src/" → list contents of src/
        let dir = if partial.starts_with('/') {
            PathBuf::from(partial)
        } else {
            cwd.join(partial)
        };
        (dir, String::new())
    } else {
        // e.g. "src/ma" → parent=src/, prefix=ma
        let path = if partial.starts_with('/') {
            PathBuf::from(partial)
        } else {
            cwd.join(partial)
        };
        let parent = path.parent().unwrap_or(&cwd).to_path_buf();
        let prefix = path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        (parent, prefix)
    };

    let entries = match std::fs::read_dir(&dir) {
        Ok(rd) => rd,
        Err(_) => return Vec::new(),
    };

    let show_hidden = prefix.starts_with('.');

    let mut dirs: Vec<(String, String)> = Vec::new();
    let mut files: Vec<(String, String)> = Vec::new();

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();

        // Skip hidden files unless prefix starts with '.'
        if !show_hidden && name.starts_with('.') {
            continue;
        }

        if !prefix.is_empty() && !name.starts_with(&prefix) {
            continue;
        }

        let is_dir = entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false);

        // Build the display path relative to the partial's directory component
        let dir_part = if partial.is_empty() {
            String::new()
        } else if partial.ends_with('/') {
            partial.to_string()
        } else if let Some(slash) = partial.rfind('/') {
            partial[..=slash].to_string()
        } else {
            String::new()
        };

        if is_dir {
            dirs.push((format!("{}{}/", dir_part, name), "dir".to_string()));
        } else {
            files.push((format!("{}{}", dir_part, name), "file".to_string()));
        }
    }

    dirs.sort_by(|a, b| a.0.cmp(&b.0));
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let mut result = dirs;
    result.extend(files);
    result.truncate(20);
    result
}

/// Compute the visual height of pending streaming text.
/// Format a token count for display: 1234 → "1.2k", 567 → "567", 12345 → "12.3k"
pub fn format_tokens(n: i64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

/// Walk up from a directory looking for a `.git/` directory.
fn find_git_root(start: &std::path::Path) -> Option<PathBuf> {
    let mut dir = start.to_path_buf();
    loop {
        if dir.join(".git").exists() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── wrapped_line_height ──

    #[test]
    fn wrapped_line_height_empty() {
        // Empty line with newline has width 0 → returns 1
        assert_eq!(wrapped_line_height("", 80), 1);
    }

    #[test]
    fn wrapped_line_height_short() {
        assert_eq!(wrapped_line_height("hello", 80), 1);
    }

    #[test]
    fn wrapped_line_height_exact_width() {
        assert_eq!(wrapped_line_height("abcd", 4), 1);
    }

    #[test]
    fn wrapped_line_height_one_over() {
        assert_eq!(wrapped_line_height("abcde", 4), 2);
    }

    #[test]
    fn wrapped_line_height_zero_width() {
        assert_eq!(wrapped_line_height("hello", 0), 1);
    }

    #[test]
    fn wrapped_line_height_cjk() {
        // CJK characters take 2 columns each
        // 3 CJK chars = 6 columns, width 4 → ceil(6/4) = 2
        assert_eq!(wrapped_line_height("\u{4e16}\u{754c}\u{597d}", 4), 2);
    }

    // ── wrapped_text_height ──

    #[test]
    fn wrapped_text_height_single_line() {
        assert_eq!(wrapped_text_height("hello", 80, 0), 1);
    }

    #[test]
    fn wrapped_text_height_multi_line() {
        assert_eq!(wrapped_text_height("line1\nline2\nline3", 80, 0), 3);
    }

    #[test]
    fn wrapped_text_height_empty() {
        assert_eq!(wrapped_text_height("", 80, 0), 1);
    }

    #[test]
    fn wrapped_text_height_with_prefix() {
        // "hello" = 5 chars, width 6, prefix 2 → effective width 4 → ceil(5/4) = 2
        assert_eq!(wrapped_text_height("hello", 6, 2), 2);
    }

    // ── format_tokens ──

    #[test]
    fn format_tokens_small() {
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(999), "999");
    }

    #[test]
    fn format_tokens_thousands() {
        assert_eq!(format_tokens(1000), "1.0k");
        assert_eq!(format_tokens(1234), "1.2k");
        assert_eq!(format_tokens(999999), "1000.0k");
    }

    #[test]
    fn format_tokens_millions() {
        assert_eq!(format_tokens(1_000_000), "1.0M");
        assert_eq!(format_tokens(2_500_000), "2.5M");
    }

    // ── DisplayBlock::height ──

    #[test]
    fn display_block_tool_call_height() {
        let block = DisplayBlock::ToolCall {
            name: "read_file".into(),
            success: true,
            duration_ms: 50,
        };
        assert_eq!(block.height(false, 80, 0), 1);
    }

    #[test]
    fn display_block_error_height() {
        let block = DisplayBlock::Error("short error".into());
        assert_eq!(block.height(false, 80, 0), 1);
    }

    #[test]
    fn display_block_user_input_height() {
        // "hello" with 2-char prefix = 1 line + 1 blank after = 2
        let block = DisplayBlock::UserInput("hello".into());
        assert_eq!(block.height(false, 80, 0), 2);
    }

    #[test]
    fn display_block_task_list_height() {
        let tasks = vec![
            TaskSnapshot {
                id: "01".into(),
                label: "A".into(),
                status: "pending".into(),
                owner: String::new(),
                is_blocked: false,
            },
            TaskSnapshot {
                id: "02".into(),
                label: "B".into(),
                status: "pending".into(),
                owner: String::new(),
                is_blocked: false,
            },
        ];
        let block = DisplayBlock::TaskList { tasks };
        assert_eq!(block.height(false, 80, 0), 4); // top border + 2 tasks + bottom border
    }

    #[test]
    fn display_block_splash_height() {
        let block = DisplayBlock::Splash;
        assert_eq!(block.height(false, 80, 24), 24);
    }
}

/// Generate a readable slug like "swift-falcon" from small word lists.
fn generate_plan_slug() -> String {
    const ADJECTIVES: &[&str] = &[
        "swift", "bold", "calm", "dark", "keen", "warm", "cool", "fair", "deep", "wild", "soft",
        "pure", "vast", "slim", "rare", "firm", "lean", "rich", "true", "wise", "fast", "safe",
        "full", "neat", "open", "flat", "pale", "dry", "raw", "new",
    ];
    const NOUNS: &[&str] = &[
        "falcon", "ember", "coral", "prism", "ridge", "cedar", "bloom", "frost", "stone", "grain",
        "drift", "spark", "forge", "shade", "crest", "brook", "flint", "moss", "peak", "dust",
        "glow", "wave", "pine", "iron", "salt", "bone", "mist", "clay", "sage", "arch",
    ];

    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    std::time::SystemTime::now().hash(&mut hasher);
    std::process::id().hash(&mut hasher);
    let h = hasher.finish();

    let adj = ADJECTIVES[(h as usize) % ADJECTIVES.len()];
    let noun = NOUNS[((h >> 16) as usize) % NOUNS.len()];
    format!("{}-{}", adj, noun)
}
