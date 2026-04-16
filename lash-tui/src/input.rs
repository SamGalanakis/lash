use crossterm::event::{
    Event as TermEvent, KeyCode as TermKeyCode, KeyEvent as TermKeyEvent,
    KeyEventKind as TermKeyEventKind, KeyModifiers as TermKeyModifiers,
    MouseButton as TermMouseButton, MouseEvent as TermMouseEvent,
    MouseEventKind as TermMouseEventKind,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TermCapabilities {
    pub bracketed_paste: bool,
    pub focus_change: bool,
    pub mouse_capture: bool,
    pub keyboard_enhancement: bool,
    pub synchronized_update: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeyModifiers {
    pub shift: bool,
    pub control: bool,
    pub alt: bool,
    pub super_key: bool,
}

impl From<TermKeyModifiers> for KeyModifiers {
    fn from(value: TermKeyModifiers) -> Self {
        Self {
            shift: value.contains(TermKeyModifiers::SHIFT),
            control: value.contains(TermKeyModifiers::CONTROL),
            alt: value.contains(TermKeyModifiers::ALT),
            super_key: value.contains(TermKeyModifiers::SUPER),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyCode {
    Backspace,
    Enter,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    PageUp,
    PageDown,
    Tab,
    BackTab,
    Delete,
    Insert,
    Esc,
    F(u8),
    Char(char),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyEventKind {
    Press,
    Repeat,
    Release,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyEvent {
    pub code: KeyCode,
    pub modifiers: KeyModifiers,
    pub kind: KeyEventKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MouseEventKind {
    Down(MouseButton),
    Up(MouseButton),
    Drag(MouseButton),
    Moved,
    ScrollUp,
    ScrollDown,
    ScrollLeft,
    ScrollRight,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MouseEvent {
    pub kind: MouseEventKind,
    pub column: u16,
    pub row: u16,
    pub modifiers: KeyModifiers,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputEvent {
    Key(KeyEvent),
    Mouse(MouseEvent),
    Resize { width: u16, height: u16 },
    FocusGained,
    FocusLost,
    Paste(String),
    Tick,
}

pub fn normalize_event(event: &TermEvent) -> Option<InputEvent> {
    match event {
        TermEvent::Key(key) => Some(InputEvent::Key(normalize_key_event(*key)?)),
        TermEvent::Mouse(mouse) => Some(InputEvent::Mouse(normalize_mouse_event(*mouse)?)),
        TermEvent::Resize(width, height) => Some(InputEvent::Resize {
            width: *width,
            height: *height,
        }),
        TermEvent::FocusGained => Some(InputEvent::FocusGained),
        TermEvent::FocusLost => Some(InputEvent::FocusLost),
        TermEvent::Paste(text) => Some(InputEvent::Paste(text.clone())),
    }
}

pub fn normalize_key_event(event: TermKeyEvent) -> Option<KeyEvent> {
    let code = match event.code {
        TermKeyCode::Backspace => KeyCode::Backspace,
        TermKeyCode::Enter => KeyCode::Enter,
        TermKeyCode::Left => KeyCode::Left,
        TermKeyCode::Right => KeyCode::Right,
        TermKeyCode::Up => KeyCode::Up,
        TermKeyCode::Down => KeyCode::Down,
        TermKeyCode::Home => KeyCode::Home,
        TermKeyCode::End => KeyCode::End,
        TermKeyCode::PageUp => KeyCode::PageUp,
        TermKeyCode::PageDown => KeyCode::PageDown,
        TermKeyCode::Tab => KeyCode::Tab,
        TermKeyCode::BackTab => KeyCode::BackTab,
        TermKeyCode::Delete => KeyCode::Delete,
        TermKeyCode::Insert => KeyCode::Insert,
        TermKeyCode::Esc => KeyCode::Esc,
        TermKeyCode::F(value) => KeyCode::F(value),
        TermKeyCode::Char(ch) => KeyCode::Char(ch),
        _ => return None,
    };

    let kind = match event.kind {
        TermKeyEventKind::Press => KeyEventKind::Press,
        TermKeyEventKind::Repeat => KeyEventKind::Repeat,
        TermKeyEventKind::Release => KeyEventKind::Release,
    };

    Some(KeyEvent {
        code,
        modifiers: event.modifiers.into(),
        kind,
    })
}

pub fn normalize_mouse_event(event: TermMouseEvent) -> Option<MouseEvent> {
    let button = |button| match button {
        TermMouseButton::Left => MouseButton::Left,
        TermMouseButton::Right => MouseButton::Right,
        TermMouseButton::Middle => MouseButton::Middle,
    };
    let kind = match event.kind {
        TermMouseEventKind::Down(button_kind) => MouseEventKind::Down(button(button_kind)),
        TermMouseEventKind::Up(button_kind) => MouseEventKind::Up(button(button_kind)),
        TermMouseEventKind::Drag(button_kind) => MouseEventKind::Drag(button(button_kind)),
        TermMouseEventKind::Moved => MouseEventKind::Moved,
        TermMouseEventKind::ScrollUp => MouseEventKind::ScrollUp,
        TermMouseEventKind::ScrollDown => MouseEventKind::ScrollDown,
        TermMouseEventKind::ScrollLeft => MouseEventKind::ScrollLeft,
        TermMouseEventKind::ScrollRight => MouseEventKind::ScrollRight,
    };

    Some(MouseEvent {
        kind,
        column: event.column,
        row: event.row,
        modifiers: event.modifiers.into(),
    })
}
