pub fn draw(frame: &mut Frame<'_>, app: &mut App) {
    draw_with_capabilities(frame, app, TermCapabilities::default());
}

pub fn draw_with_capabilities(
    frame: &mut Frame<'_>,
    app: &mut App,
    capabilities: TermCapabilities,
) {
    let area = frame.area();
    if area.width == 0 || area.height == 0 {
        return;
    }

    frame.clear(bg(theme::surface_base()));

    // One layout pass for the whole frame; the `render::*_area` accessors each
    // recompute `chrome_layout`, so calling them per region would repeat that
    // work five times per draw.
    let render::ChromeAreas {
        status: status_area,
        history,
        dock: dock_area,
        queue: queue_area,
        footer: footer_area,
        input: input_area,
        process: process_area,
        body: body_area,
    } = render::chrome_areas(app, area.width, area.height);
    let queue_lines = render::queue_preview_lines_snapshot(app, area.width);

    let surfaces = app.ui_extensions().surface_scene();
    draw_status_bar(frame, app, status_area);
    let surfaces = sync_surface_areas(app, surfaces, history, dock_area, footer_area, body_area);
    if surfaces.has_slot(TuiSurfaceSlot::Workspace) {
        draw_workspace_surface(frame, app, &surfaces, history, capabilities);
    } else {
        draw_history(frame, app, history);
        apply_selection_highlight(frame, app, history);
    }
    if dock_area.height > 0 {
        draw_surface_stack(
            frame,
            app,
            &surfaces,
            TuiSurfaceSlot::Dock,
            dock_area,
            capabilities,
        );
    }
    if queue_area.height > 0 {
        draw_lines_region(frame, queue_area, &queue_lines, bg(theme::surface_raised()));
    }
    if footer_area.height > 0 {
        draw_surface_stack(
            frame,
            app,
            &surfaces,
            TuiSurfaceSlot::Footer,
            footer_area,
            capabilities,
        );
    }
    if app.has_prompt() {
        draw_overlay_scrim(frame, history);
        draw_prompt(frame, app, input_area);
    } else {
        draw_input(frame, app, input_area);
        draw_suggestions(frame, app, input_area);
    }
    draw_process_dock(frame, app, process_area);
    draw_session_picker(frame, app, history);
    draw_tree(frame, app, history);
    draw_skill_picker(frame, app, history);
    draw_process_overview(frame, app, body_area);
    draw_document_overlay(frame, app, body_area);
    draw_overlay_surface(frame, app, &surfaces, body_area, capabilities);
}

fn sync_surface_areas(
    app: &App,
    mut surfaces: TuiSurfaceScene,
    history_area: Rect,
    dock_area: Rect,
    footer_area: Rect,
    body_area: Rect,
) -> TuiSurfaceScene {
    let mut assignments = Vec::new();
    if let Some(surface) = surfaces.workspace.last_mut() {
        surface.area = Some(history_area);
        assignments.push((surface.id.clone(), history_area));
    }
    let mut dock_y = dock_area.y;
    let mut dock_remaining = dock_area.height;
    for surface in &mut surfaces.dock {
        if dock_remaining == 0 {
            break;
        }
        let height = surface.size.height().min(dock_remaining);
        if height == 0 {
            continue;
        }
        let area = Rect::new(dock_area.x, dock_y, dock_area.width, height);
        surface.area = Some(area);
        assignments.push((surface.id.clone(), area));
        dock_y = dock_y.saturating_add(height);
        dock_remaining = dock_remaining.saturating_sub(height);
    }
    let mut footer_y = footer_area.y;
    let mut footer_remaining = footer_area.height;
    for surface in &mut surfaces.footer {
        if footer_remaining == 0 {
            break;
        }
        let height = surface.size.height().min(footer_remaining);
        if height == 0 {
            continue;
        }
        let area = Rect::new(footer_area.x, footer_y, footer_area.width, height);
        surface.area = Some(area);
        assignments.push((surface.id.clone(), area));
        footer_y = footer_y.saturating_add(height);
        footer_remaining = footer_remaining.saturating_sub(height);
    }
    if let Some(surface) = surfaces.overlay.last_mut()
        && body_area.width > 0
        && body_area.height > 0
    {
        let width = surface
            .size
            .width()
            .unwrap_or_else(|| body_area.width.saturating_sub(4).max(1))
            .min(body_area.width);
        let height = surface.size.height().min(body_area.height).max(1);
        let x = body_area.x + body_area.width.saturating_sub(width) / 2;
        let y = body_area.y + body_area.height.saturating_sub(height) / 2;
        let area = Rect::new(x, y, width, height);
        surface.area = Some(area);
        assignments.push((surface.id.clone(), area));
    }
    app.ui_extensions().sync_surface_areas(assignments);
    surfaces
}

fn draw_workspace_surface(
    frame: &mut Frame<'_>,
    app: &App,
    surfaces: &TuiSurfaceScene,
    area: Rect,
    capabilities: TermCapabilities,
) {
    let Some(surface) = surfaces.workspace.last() else {
        return;
    };
    let mut viewport = frame.viewport(area);
    app.ui_extensions().render_mounted_surface(
        surface,
        TuiRenderContext {
            session_id: app.session_id.as_str(),
            capabilities,
            surface_id: &surface.id,
            focused: surfaces.focused.as_deref() == Some(surface.id.as_str()),
        },
        &mut viewport,
    );
}

fn draw_surface_stack(
    frame: &mut Frame<'_>,
    app: &App,
    surfaces: &TuiSurfaceScene,
    slot: TuiSurfaceSlot,
    area: Rect,
    capabilities: TermCapabilities,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    for surface in surfaces.surfaces(slot) {
        let Some(surface_area) = surface.area else {
            continue;
        };
        let mut viewport = frame.viewport(surface_area);
        viewport.clear(bg(theme::surface_raised()));
        app.ui_extensions().render_mounted_surface(
            surface,
            TuiRenderContext {
                session_id: app.session_id.as_str(),
                capabilities,
                surface_id: &surface.id,
                focused: surfaces.focused.as_deref() == Some(surface.id.as_str()),
            },
            &mut viewport,
        );
    }
}

fn draw_overlay_surface(
    frame: &mut Frame<'_>,
    app: &App,
    surfaces: &TuiSurfaceScene,
    area: Rect,
    capabilities: TermCapabilities,
) {
    let Some(surface) = surfaces.overlay.last() else {
        return;
    };
    let Some(surface_area) = surface.area else {
        return;
    };
    draw_overlay_scrim(frame, area);
    let mut viewport = frame.viewport(surface_area);
    viewport.clear(bg(theme::surface_base()));
    app.ui_extensions().render_mounted_surface(
        surface,
        TuiRenderContext {
            session_id: app.session_id.as_str(),
            capabilities,
            surface_id: &surface.id,
            focused: surfaces.focused.as_deref() == Some(surface.id.as_str()),
        },
        &mut viewport,
    );
}

// ─── Status bar slot grammar ─────────────────────────────────────────────────
//
// The status bar is built from labeled slots, each with a `priority`. When
// the window is too narrow to fit every slot, the lowest-priority slots are
// dropped first until the remainder fits. There is no character-level
// truncation: a slot either renders in full or not at all. This keeps the
// bar's shape legible at every width and makes it obvious which information
// is load-bearing (brand, model) versus decorative (variant, context meter).
