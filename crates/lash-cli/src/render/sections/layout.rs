fn padded_content_width(frame_width: u16, horizontal_padding: u16) -> usize {
    frame_width.saturating_sub(horizontal_padding.saturating_mul(2)) as usize
}

fn prompt_inner_width(frame_width: u16) -> usize {
    frame_width.saturating_sub(PROMPT_HORIZONTAL_PADDING.saturating_mul(2)) as usize
}

fn desired_input_height(app: &App, frame_width: u16) -> u16 {
    if app.has_prompt() {
        prompt_height(app, frame_width, u16::MAX)
    } else {
        let inner_w = padded_content_width(frame_width, INPUT_HORIZONTAL_PADDING);
        let visual_lines = input_visual_lines(app.input(), inner_w);
        (visual_lines as u16 + 2).min(MAX_INPUT_HEIGHT)
    }
}

fn input_height(app: &App, frame_width: u16, frame_height: u16, reserved_height: u16) -> u16 {
    let desired = desired_input_height(app, frame_width);
    let max_allowed = frame_height
        .saturating_sub(reserved_height)
        .saturating_sub(MIN_HISTORY_HEIGHT);
    desired.min(max_allowed)
}

fn queue_preview_height(app: &App, frame_width: u16) -> u16 {
    queue_preview_lines_snapshot(app, frame_width).len() as u16
}

/// Rows the plan checklist contributes to the trailing end of the
/// transcript: 1 blank gutter + 1 `Plan` header + N items. Must stay in
/// lockstep with [`plan_dock_lines_snapshot`](crate::render::plan_dock_lines_snapshot),
/// since the bottom-anchor pad derives the underflow from this total.
pub fn plan_dock_trailing_height(app: &App) -> usize {
    match app.plan_dock.as_ref() {
        Some(plan) if !plan.is_empty() => 2 + plan.items.len(),
        _ => 0,
    }
}

pub fn process_dock_height(app: &App, frame_width: u16) -> u16 {
    process_lines_snapshot(app, frame_width)
        .map(|lines| lines.len() as u16)
        .unwrap_or(0)
}

#[derive(Clone, Copy, Debug)]
struct ChromeLayout {
    history_height: u16,
    dock_height: u16,
    queue_height: u16,
    footer_height: u16,
    input_height: u16,
    process_height: u16,
}

fn chrome_layout(app: &App, frame_width: u16, frame_height: u16) -> ChromeLayout {
    let surfaces = app.ui_extensions().surface_scene();
    let queue_height = queue_preview_height(app, frame_width);
    let footer_available = frame_height
        .saturating_sub(1 + queue_height)
        .saturating_sub(MIN_HISTORY_HEIGHT);
    let footer_height = surfaces.stack_height(TuiSurfaceSlot::Footer, footer_available);
    let dock_available = frame_height
        .saturating_sub(1 + queue_height + footer_height)
        .saturating_sub(MIN_HISTORY_HEIGHT);
    let dock_height = surfaces.stack_height(TuiSurfaceSlot::Dock, dock_available);
    let input_desired = desired_input_height(app, frame_width);
    let process_desired = process_dock_height(app, frame_width);
    let process_available = frame_height
        .saturating_sub(1 + dock_height + queue_height + footer_height)
        .saturating_sub(input_desired)
        .saturating_sub(MIN_HISTORY_HEIGHT);
    let process_height = process_desired.min(process_available);
    // Reserve one status row, rendered below the input, for model/context/git
    // metadata. This replaces the old top chrome row so the transcript can
    // start at row 0 and all session metadata lives at the bottom.
    let reserved_height = 1 + dock_height + queue_height + footer_height + process_height;
    let input_height = input_height(app, frame_width, frame_height, reserved_height);
    let history_height = frame_height.saturating_sub(
        1 + dock_height + queue_height + footer_height + input_height + process_height,
    );
    ChromeLayout {
        history_height,
        dock_height,
        queue_height,
        footer_height,
        input_height,
        process_height,
    }
}

/// Every chrome region for one frame, derived from a single `chrome_layout`
/// pass. The frame draw uses this instead of calling `history_area`,
/// `dock_area`, … separately, each of which recomputes the whole layout.
pub struct ChromeAreas {
    pub status: Rect,
    pub history: Rect,
    pub dock: Rect,
    pub queue: Rect,
    pub footer: Rect,
    pub input: Rect,
    pub process: Rect,
    pub body: Rect,
}

/// Compute every chrome region in one `chrome_layout` pass. The individual
/// `*_area` accessors below stay for callers that need a single region; the
/// draw loop uses this to avoid recomputing the layout once per region.
pub fn chrome_areas(app: &App, frame_width: u16, frame_height: u16) -> ChromeAreas {
    let layout = chrome_layout(app, frame_width, frame_height);
    let history_y = 0;
    let dock_y = history_y + layout.history_height;
    let queue_y = dock_y + layout.dock_height;
    let footer_y = queue_y + layout.queue_height;
    let input_y = footer_y + layout.footer_height;
    let status_y = input_y + layout.input_height;
    let process_y = status_y + 1;
    ChromeAreas {
        status: Rect::new(0, status_y, frame_width, 1),
        history: Rect::new(0, history_y, frame_width, layout.history_height),
        dock: Rect::new(0, dock_y, frame_width, layout.dock_height),
        queue: Rect::new(0, queue_y, frame_width, layout.queue_height),
        footer: Rect::new(0, footer_y, frame_width, layout.footer_height),
        input: Rect::new(0, input_y, frame_width, layout.input_height),
        process: Rect::new(0, process_y, frame_width, layout.process_height),
        body: Rect::new(0, 0, frame_width, frame_height),
    }
}

pub fn history_viewport_height(app: &App, frame_width: u16, frame_height: u16) -> usize {
    chrome_layout(app, frame_width, frame_height).history_height as usize
}

pub fn history_area(app: &App, frame_width: u16, frame_height: u16) -> Rect {
    let layout = chrome_layout(app, frame_width, frame_height);
    Rect::new(0, 0, frame_width, layout.history_height)
}

pub fn input_area(app: &App, frame_width: u16, frame_height: u16) -> Rect {
    let layout = chrome_layout(app, frame_width, frame_height);
    let y = layout.history_height + layout.dock_height + layout.queue_height + layout.footer_height;
    Rect::new(0, y, frame_width, layout.input_height)
}

pub fn input_content_area(app: &App, frame_width: u16, frame_height: u16) -> Rect {
    input_content_area_from_frame(input_area(app, frame_width, frame_height))
}

fn input_content_area_from_frame(area: Rect) -> Rect {
    Rect::new(
        area.x + INPUT_HORIZONTAL_PADDING,
        area.y + 1,
        area.width.saturating_sub(INPUT_HORIZONTAL_PADDING * 2),
        area.height.saturating_sub(2),
    )
}
