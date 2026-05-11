use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lash::TurnEvent;
use lash::control::PluginAction;
use lash_tui::{
    Axis, Color, Column, ColumnWidth, Constraint, Frame, InputEvent, KeyCode as InputKeyCode,
    KeyEventKind, Layout, Line, Modifier, Rect, Span, Style, Table, TableCell, TableRow,
    TableState,
};
use lash_tui_extensions::{
    KeyChord, KeyCode, KeyModifiers, ShortcutSpec, SlashCommandSpec, TuiExtension,
    TuiExtensionContext, TuiHostEffect, TuiInputOutcome, TuiRenderContext, TuiSurfaceSize,
    TuiSurfaceSlot, TuiSurfaceSpec, call_plugin_action,
};
use serde_json::Value;

use crate::model::{
    Direction, ExperimentStatus, PLUGIN_ID, RunningStatus, StatusSummary, format_confidence,
    format_delta, format_metric, format_seconds,
};

const WORKSPACE_KEY: &str = "workspace";
const FOOTER_KEY: &str = "footer";
const OVERLAY_KEY: &str = "overlay";
const AUTORESEARCH_ARGUMENT_HINT: &str = "[objective|help|off|clear|export]";

const AUTORESEARCH_COMMANDS: &[SlashCommandSpec] = &[SlashCommandSpec {
    name: "/autoresearch",
    aliases: &[],
    usage: "/autoresearch [objective|help|off|clear|export]",
    description: "Toggle or control autoresearch mode",
    argument_hint: Some(AUTORESEARCH_ARGUMENT_HINT),
    argument_options: &["help", "off", "clear", "export"],
    takes_argument: true,
    allow_while_running: true,
    action: "command",
}];

const AUTORESEARCH_SHORTCUTS: &[ShortcutSpec] = &[
    ShortcutSpec {
        chord: KeyChord {
            code: KeyCode::Char('x'),
            modifiers: KeyModifiers {
                shift: false,
                control: true,
                alt: false,
            },
        },
        description: "Toggle autoresearch table",
        action: "toggle_workspace",
    },
    ShortcutSpec {
        chord: KeyChord {
            code: KeyCode::Char('x'),
            modifiers: KeyModifiers {
                shift: true,
                control: true,
                alt: false,
            },
        },
        description: "Toggle autoresearch overlay",
        action: "toggle_overlay",
    },
];

#[derive(Clone, Default)]
pub struct AutoresearchTuiExtension {
    state: Arc<Mutex<SessionUiState>>,
}

#[derive(Clone, Default)]
struct SessionUiState {
    summary: StatusSummary,
    expanded: bool,
    overlay_open: bool,
    workspace_table: TableState,
    overlay_table: TableState,
    workspace_rows: usize,
    overlay_rows: usize,
}

#[async_trait]
impl TuiExtension for AutoresearchTuiExtension {
    fn id(&self) -> &'static str {
        PLUGIN_ID
    }

    fn commands(&self) -> &'static [SlashCommandSpec] {
        AUTORESEARCH_COMMANDS
    }

    fn shortcuts(&self) -> &'static [ShortcutSpec] {
        AUTORESEARCH_SHORTCUTS
    }

    async fn snapshot(&self, ctx: TuiExtensionContext<'_>) -> Result<Vec<TuiHostEffect>, String> {
        let summary = fetch_status(ctx).await?;
        let mut state = self.lock_state()?;
        state.summary = summary;
        let result_len = state.summary.results.len();
        sync_selection(&mut state.workspace_table, result_len);
        sync_selection(&mut state.overlay_table, result_len);
        Ok(surface_effects(&state))
    }

    async fn invoke_action(
        &self,
        action: &str,
        arg: Option<&str>,
        ctx: TuiExtensionContext<'_>,
    ) -> Result<Vec<TuiHostEffect>, String> {
        match action {
            "command" => self.handle_command(arg, ctx).await,
            "toggle_workspace" => {
                let mut state = self.lock_state()?;
                if !state.summary.active {
                    return Ok(vec![TuiHostEffect::PushSystemMessage(
                        "Autoresearch mode is off.".to_string(),
                    )]);
                }
                state.expanded = !state.expanded;
                let mut effects = surface_effects(&state);
                if state.expanded {
                    effects.push(TuiHostEffect::FocusSurface {
                        key: WORKSPACE_KEY.to_string(),
                    });
                } else {
                    effects.push(TuiHostEffect::BlurSurface {
                        key: WORKSPACE_KEY.to_string(),
                    });
                }
                Ok(effects)
            }
            "toggle_overlay" => {
                let mut state = self.lock_state()?;
                if !state.summary.active {
                    return Ok(vec![TuiHostEffect::PushSystemMessage(
                        "Autoresearch mode is off.".to_string(),
                    )]);
                }
                state.overlay_open = !state.overlay_open;
                let mut effects = surface_effects(&state);
                if state.overlay_open {
                    effects.push(TuiHostEffect::FocusSurface {
                        key: OVERLAY_KEY.to_string(),
                    });
                } else {
                    effects.push(TuiHostEffect::BlurSurface {
                        key: OVERLAY_KEY.to_string(),
                    });
                }
                Ok(effects)
            }
            other => Err(format!("unknown autoresearch UI action `{other}`")),
        }
    }

    fn render_surface(&self, surface_key: &str, ctx: TuiRenderContext<'_>, frame: &mut Frame<'_>) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        match surface_key {
            FOOTER_KEY => render_footer_surface(&state, frame),
            WORKSPACE_KEY => render_workspace_surface(&mut state, ctx, frame),
            OVERLAY_KEY => render_overlay_surface(&mut state, ctx, frame),
            _ => {}
        }
    }

    fn handle_surface_input(
        &self,
        surface_key: &str,
        event: &InputEvent,
        _ctx: TuiExtensionContext<'_>,
    ) -> TuiInputOutcome {
        let Ok(mut state) = self.state.lock() else {
            return TuiInputOutcome::Ignored;
        };
        let result_len = state.summary.results.len();
        let workspace_rows = state.workspace_rows.max(1);
        let overlay_rows = state.overlay_rows.max(1);
        let handled = match surface_key {
            WORKSPACE_KEY => handle_table_input(
                event,
                &mut state.workspace_table,
                result_len,
                workspace_rows,
                false,
            ),
            OVERLAY_KEY => handle_table_input(
                event,
                &mut state.overlay_table,
                result_len,
                overlay_rows,
                true,
            ),
            _ => TableInputOutcome::Ignored,
        };
        match handled {
            TableInputOutcome::Ignored => TuiInputOutcome::Ignored,
            TableInputOutcome::Handled => TuiInputOutcome::Handled(Vec::new()),
            TableInputOutcome::CloseOverlay => {
                state.overlay_open = false;
                TuiInputOutcome::Handled(vec![
                    TuiHostEffect::BlurSurface {
                        key: OVERLAY_KEY.to_string(),
                    },
                    TuiHostEffect::UnmountSurface {
                        key: OVERLAY_KEY.to_string(),
                    },
                ])
            }
        }
    }

    fn handle_turn_event(&self, event: &TurnEvent) -> Vec<TuiHostEffect> {
        let TurnEvent::PluginSurface { plugin_id, event } = event else {
            return Vec::new();
        };
        if plugin_id != PLUGIN_ID {
            return Vec::new();
        }
        let lash_core::PluginSurfaceEvent::Custom { name, payload } = event else {
            return Vec::new();
        };
        if name != "autoresearch.status" {
            return Vec::new();
        }
        let Ok(summary) = serde_json::from_value::<StatusSummary>(payload.clone()) else {
            return Vec::new();
        };
        let mut state = match self.state.lock() {
            Ok(value) => value,
            Err(_) => return Vec::new(),
        };
        state.summary = summary;
        let result_len = state.summary.results.len();
        sync_selection(&mut state.workspace_table, result_len);
        sync_selection(&mut state.overlay_table, result_len);
        surface_effects(&state)
    }
}

impl AutoresearchTuiExtension {
    async fn handle_command(
        &self,
        arg: Option<&str>,
        ctx: TuiExtensionContext<'_>,
    ) -> Result<Vec<TuiHostEffect>, String> {
        match arg.map(str::trim).filter(|value| !value.is_empty()) {
            Some(raw) if raw.eq_ignore_ascii_case("help") => {
                let state = self.lock_state()?;
                let mut effects = surface_effects(&state);
                effects.push(TuiHostEffect::PushSystemMessage(autoresearch_help_text()));
                Ok(effects)
            }
            Some(raw) if raw.eq_ignore_ascii_case("off") => {
                let result = invoke_plugin_action_op::<crate::AutoresearchStopOp>(
                    ctx,
                    crate::AutoresearchEmptyArgs {},
                )
                .await?;
                self.apply_command_result(result, false)
            }
            Some(raw) if raw.eq_ignore_ascii_case("clear") => {
                let result = invoke_plugin_action_op::<crate::AutoresearchClearOp>(
                    ctx,
                    crate::AutoresearchEmptyArgs {},
                )
                .await?;
                self.apply_command_result(result, true)
            }
            Some(raw) if raw.eq_ignore_ascii_case("export") => {
                let result = call_plugin_action::<crate::AutoresearchExportOp>(
                    ctx.actions,
                    crate::AutoresearchEmptyArgs {},
                )
                .await?;
                let status = result.status;
                let path = result.path;
                let mut state = self.lock_state()?;
                state.summary = status;
                let mut effects = surface_effects(&state);
                effects.push(TuiHostEffect::PushSystemMessage(if path.is_empty() {
                    "Exported autoresearch summary.".to_string()
                } else {
                    format!("Wrote {path}.")
                }));
                Ok(effects)
            }
            objective => {
                let result = invoke_plugin_action_op::<crate::AutoresearchStartOp>(
                    ctx,
                    crate::AutoresearchStartArgs {
                        objective: objective.map(str::to_string),
                    },
                )
                .await?;
                self.apply_command_result(result, false)
            }
        }
    }

    fn apply_command_result(
        &self,
        result: Value,
        clear_ui: bool,
    ) -> Result<Vec<TuiHostEffect>, String> {
        let status = parse_status_field(&result)?;
        let message = result
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("Autoresearch updated.")
            .to_string();
        let queued_input = result
            .get("queued_input")
            .and_then(Value::as_str)
            .map(str::to_string);
        let mut state = self.lock_state()?;
        state.summary = status;
        if clear_ui || !state.summary.active {
            state.expanded = false;
            state.overlay_open = false;
        }
        let result_len = state.summary.results.len();
        sync_selection(&mut state.workspace_table, result_len);
        sync_selection(&mut state.overlay_table, result_len);
        let mut effects = surface_effects(&state);
        effects.push(TuiHostEffect::PushSystemMessage(message));
        if let Some(input) = queued_input {
            effects.push(TuiHostEffect::QueueTurn { input });
        }
        Ok(effects)
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, SessionUiState>, String> {
        self.state
            .lock()
            .map_err(|_| "autoresearch UI state poisoned".to_string())
    }
}

fn autoresearch_help_text() -> String {
    [
        "Autoresearch",
        "",
        "Commands:",
        "  /autoresearch <objective>  Start autoresearch or update the objective",
        "  /autoresearch help         Show this help",
        "  /autoresearch off          Stop autoresearch mode",
        "  /autoresearch clear        Clear autoresearch state and UI",
        "  /autoresearch export       Export the current summary",
        "",
        "Shortcuts:",
        "  Ctrl+X         Toggle the autoresearch table",
        "  Ctrl+Shift+X   Toggle the autoresearch overlay",
        "",
        "Workflow:",
        "  1. Start with /autoresearch <objective>.",
        "  2. Let the agent iterate with init_experiment, run_experiment, and log_experiment.",
        "  3. Use export when you want a written summary artifact.",
    ]
    .join("\n")
}

async fn fetch_status(ctx: TuiExtensionContext<'_>) -> Result<StatusSummary, String> {
    call_plugin_action::<crate::AutoresearchStatusOp>(ctx.actions, crate::AutoresearchEmptyArgs {})
        .await
}

async fn invoke_plugin_action_op<Op>(
    ctx: TuiExtensionContext<'_>,
    args: Op::Args,
) -> Result<Value, String>
where
    Op: PluginAction,
    Op::Output: serde::Serialize,
{
    let result = call_plugin_action::<Op>(ctx.actions, args).await?;
    serde_json::to_value(result).map_err(|err| format!("invalid autoresearch output: {err}"))
}

fn parse_status_field(value: &Value) -> Result<StatusSummary, String> {
    let status = value
        .get("status")
        .cloned()
        .unwrap_or_else(|| value.clone());
    serde_json::from_value(status)
        .map_err(|err| format!("invalid autoresearch status payload: {err}"))
}

fn surface_effects(state: &SessionUiState) -> Vec<TuiHostEffect> {
    let mut effects = Vec::new();
    if state.summary.active {
        effects.push(TuiHostEffect::MountSurface {
            spec: TuiSurfaceSpec {
                key: FOOTER_KEY.to_string(),
                slot: TuiSurfaceSlot::Footer,
                size: TuiSurfaceSize::Lines(1),
                order: 20,
                focusable: false,
                visible: true,
                modal: false,
            },
        });
        if state.expanded {
            effects.push(TuiHostEffect::MountSurface {
                spec: TuiSurfaceSpec {
                    key: WORKSPACE_KEY.to_string(),
                    slot: TuiSurfaceSlot::Workspace,
                    size: TuiSurfaceSize::Auto,
                    order: 0,
                    focusable: true,
                    visible: true,
                    modal: false,
                },
            });
        } else {
            effects.push(TuiHostEffect::UnmountSurface {
                key: WORKSPACE_KEY.to_string(),
            });
        }
        if state.overlay_open {
            effects.push(TuiHostEffect::MountSurface {
                spec: TuiSurfaceSpec {
                    key: OVERLAY_KEY.to_string(),
                    slot: TuiSurfaceSlot::Overlay,
                    size: TuiSurfaceSize::Fixed {
                        width: 100,
                        height: 24,
                    },
                    order: 50,
                    focusable: true,
                    visible: true,
                    modal: true,
                },
            });
        } else {
            effects.push(TuiHostEffect::UnmountSurface {
                key: OVERLAY_KEY.to_string(),
            });
        }
    } else {
        effects.push(TuiHostEffect::UnmountSurface {
            key: FOOTER_KEY.to_string(),
        });
        effects.push(TuiHostEffect::UnmountSurface {
            key: WORKSPACE_KEY.to_string(),
        });
        effects.push(TuiHostEffect::UnmountSurface {
            key: OVERLAY_KEY.to_string(),
        });
    }
    effects
}

fn sync_selection(table: &mut TableState, len: usize) {
    table.selection.clamp(len);
    if table.selection.selected.is_none() && len > 0 {
        table.selection.select(Some(0));
    }
}

fn render_footer_surface(state: &SessionUiState, frame: &mut Frame<'_>) {
    let area = local_area(frame);
    frame.fill(area, ' ', footer_bg());
    let line = footer_line(&state.summary);
    frame.write_line_styled(0, 0, &line, footer_bg(), area.width);
}

fn render_workspace_surface(
    state: &mut SessionUiState,
    ctx: TuiRenderContext<'_>,
    frame: &mut Frame<'_>,
) {
    let area = local_area(frame);
    frame.fill(area, ' ', surface_bg());
    let parts = Layout::split(
        area,
        Axis::Vertical,
        &[
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Fill(1),
        ],
    );
    frame.write_line_styled(0, 0, &title_line(&state.summary), surface_bg(), area.width);
    frame.write_line_styled(
        0,
        1,
        &summary_line(&state.summary),
        surface_bg(),
        area.width,
    );
    if let Some(table_area) = parts.get(2).copied() {
        state.workspace_rows = table_area.height.saturating_sub(1) as usize;
        let mut viewport = frame.viewport(table_area);
        render_results_table(
            &state.summary,
            &mut state.workspace_table,
            ctx.focused,
            &mut viewport,
        );
    }
}

fn render_overlay_surface(
    state: &mut SessionUiState,
    ctx: TuiRenderContext<'_>,
    frame: &mut Frame<'_>,
) {
    let area = local_area(frame);
    frame.fill(area, ' ', overlay_bg());
    frame.draw_box(area, border_style(), Some(overlay_bg()));
    let inner = Rect::new(
        1,
        1,
        area.width.saturating_sub(2),
        area.height.saturating_sub(2),
    );
    let mut viewport = frame.viewport(inner);
    let viewport_area = local_area(&viewport);
    viewport.fill(viewport_area, ' ', overlay_bg());
    let parts = Layout::split(
        viewport_area,
        Axis::Vertical,
        &[
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Fill(1),
        ],
    );
    viewport.write_line_styled(0, 0, &title_line(&state.summary), overlay_bg(), inner.width);
    viewport.write_line_styled(
        0,
        1,
        &overlay_hint_line(&state.summary),
        overlay_bg(),
        inner.width,
    );
    if let Some(table_area) = parts.get(2).copied() {
        state.overlay_rows = table_area.height.saturating_sub(1) as usize;
        let mut table_view = viewport.viewport(table_area);
        render_results_table(
            &state.summary,
            &mut state.overlay_table,
            ctx.focused,
            &mut table_view,
        );
    }
}

fn render_results_table(
    summary: &StatusSummary,
    table_state: &mut TableState,
    focused: bool,
    frame: &mut Frame<'_>,
) {
    table_state.focused = focused;
    let table = build_table(summary);
    table.render(frame, table_state);
}

fn build_table(summary: &StatusSummary) -> Table<'static> {
    let metric_name = summary
        .metric_name
        .as_deref()
        .unwrap_or("metric")
        .to_string();
    Table {
        columns: vec![
            Column {
                header: Line::from("run"),
                width: ColumnWidth::Length(5),
            },
            Column {
                header: Line::from("commit"),
                width: ColumnWidth::Length(10),
            },
            Column {
                header: Line::from(metric_name),
                width: ColumnWidth::Length(20),
            },
            Column {
                header: Line::from("status"),
                width: ColumnWidth::Length(14),
            },
            Column {
                header: Line::from("description"),
                width: ColumnWidth::Fill(1),
            },
        ],
        rows: summary
            .results
            .iter()
            .map(|row| {
                let metric = match row.delta_percent {
                    Some(delta) => format!(
                        "{}{} {}",
                        format_metric(row.metric),
                        summary.metric_unit,
                        format_delta(delta)
                    ),
                    None => format!("{}{}", format_metric(row.metric), summary.metric_unit),
                };
                let status = row.status.label().to_string();
                let mut table_row = TableRow::new(vec![
                    TableCell {
                        content: Line::from(row.run.to_string()),
                        style: dim_fg(),
                        wrap: false,
                    },
                    TableCell {
                        content: Line::from(row.commit.clone()),
                        style: dim_fg(),
                        wrap: false,
                    },
                    TableCell {
                        content: Line::from(metric),
                        style: metric_style(summary.direction),
                        wrap: false,
                    },
                    TableCell {
                        content: Line::from(status),
                        style: status_style(row.status),
                        wrap: false,
                    },
                    TableCell {
                        content: Line::from(row.description.clone()),
                        style: text_fg(),
                        wrap: true,
                    },
                ]);
                table_row.style = row_style(row.status);
                table_row
            })
            .collect(),
        header_style: header_style(),
        row_style: surface_bg(),
        selected_row_style: selection_style(),
        focused_selected_row_style: focused_selection_style(),
    }
}

fn footer_line(summary: &StatusSummary) -> Line<'static> {
    let mut spans = vec![
        Span::styled(" autoresearch ", badge_style()),
        Span::raw(" "),
        Span::styled(
            format!("{} runs {} kept", summary.run_count, summary.kept_count),
            text_fg(),
        ),
    ];
    if let Some(running) = summary.running.as_ref() {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            format!("running {}", elapsed_since(running)),
            warning_fg(),
        ));
        if summary.run_count == 0 {
            spans.push(Span::raw("  "));
            spans.push(Span::styled("waiting for first logged result", dim_fg()));
        }
    }
    if let (Some(metric_name), Some(best_metric)) =
        (summary.metric_name.as_deref(), summary.best_metric)
    {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            format!(
                "{metric_name}: {}{}",
                format_metric(best_metric),
                summary.metric_unit
            ),
            metric_style(summary.direction),
        ));
    }
    if let Some(delta) = summary.best_delta_percent {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(format_delta(delta), accent_fg()));
    }
    if let Some(confidence) = summary.confidence {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            format!("conf {}", format_confidence(confidence)),
            confidence_style(confidence),
        ));
    }
    if let Some(last_run) = summary.last_run.as_ref() {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            format!("last {}", format_seconds(last_run.duration_seconds)),
            dim_fg(),
        ));
    }
    Line::from(spans)
}

fn title_line(summary: &StatusSummary) -> Line<'static> {
    let mut spans = vec![Span::styled("autoresearch", title_style())];
    if let Some(name) = summary.name.as_deref() {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(name.to_string(), text_fg()));
    }
    Line::from(spans)
}

fn summary_line(summary: &StatusSummary) -> Line<'static> {
    let objective = summary.objective.as_deref().unwrap_or("No objective set.");
    let status = if let Some(running) = summary.running.as_ref() {
        format!(
            "{} runs  {} kept  running {}",
            summary.run_count,
            summary.kept_count,
            elapsed_since(running)
        )
    } else {
        format!("{} runs  {} kept", summary.run_count, summary.kept_count)
    };
    Line::from(vec![
        Span::styled(objective.to_string(), text_fg()),
        Span::raw("  "),
        Span::styled(status, dim_fg()),
    ])
}

fn overlay_hint_line(summary: &StatusSummary) -> Line<'static> {
    let mut spans = vec![Span::styled(
        "Esc/q close  j/k or ↑/↓ move  u/d or PgUp/PgDn page  g/G top/bottom".to_string(),
        dim_fg(),
    )];
    if let Some(last_run) = summary.last_run.as_ref() {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            format!(
                "{} {}",
                if last_run.passed { "pass" } else { "fail" },
                format_seconds(last_run.duration_seconds)
            ),
            if last_run.passed {
                accent_fg()
            } else {
                warning_fg()
            },
        ));
    }
    Line::from(spans)
}

enum TableInputOutcome {
    Ignored,
    Handled,
    CloseOverlay,
}

fn handle_table_input(
    event: &InputEvent,
    table: &mut TableState,
    result_len: usize,
    viewport_rows: usize,
    allow_close: bool,
) -> TableInputOutcome {
    let InputEvent::Key(key) = event else {
        return TableInputOutcome::Ignored;
    };
    if key.kind == KeyEventKind::Release {
        return TableInputOutcome::Ignored;
    }
    match key.code {
        InputKeyCode::Down => {
            table.selection.move_next(result_len);
            TableInputOutcome::Handled
        }
        InputKeyCode::Up => {
            table.selection.move_prev(result_len);
            TableInputOutcome::Handled
        }
        InputKeyCode::PageDown => {
            page_selection(table, result_len, viewport_rows as isize);
            TableInputOutcome::Handled
        }
        InputKeyCode::PageUp => {
            page_selection(table, result_len, -(viewport_rows as isize));
            TableInputOutcome::Handled
        }
        InputKeyCode::Home => {
            table.selection.home(result_len);
            table.scroll.to_start();
            TableInputOutcome::Handled
        }
        InputKeyCode::End => {
            table.selection.end(result_len);
            table.scroll.to_end(result_len, viewport_rows);
            TableInputOutcome::Handled
        }
        InputKeyCode::Esc if allow_close => TableInputOutcome::CloseOverlay,
        InputKeyCode::Char('j') | InputKeyCode::Char('J') => {
            table.selection.move_next(result_len);
            TableInputOutcome::Handled
        }
        InputKeyCode::Char('k') | InputKeyCode::Char('K') => {
            table.selection.move_prev(result_len);
            TableInputOutcome::Handled
        }
        InputKeyCode::Char('d') | InputKeyCode::Char('D') => {
            page_selection(table, result_len, viewport_rows as isize);
            TableInputOutcome::Handled
        }
        InputKeyCode::Char('u') | InputKeyCode::Char('U') => {
            page_selection(table, result_len, -(viewport_rows as isize));
            TableInputOutcome::Handled
        }
        InputKeyCode::Char('g') => {
            table.selection.home(result_len);
            table.scroll.to_start();
            TableInputOutcome::Handled
        }
        InputKeyCode::Char('G') => {
            table.selection.end(result_len);
            table.scroll.to_end(result_len, viewport_rows);
            TableInputOutcome::Handled
        }
        InputKeyCode::Char('q') | InputKeyCode::Char('Q') if allow_close => {
            TableInputOutcome::CloseOverlay
        }
        _ => TableInputOutcome::Ignored,
    }
}

fn elapsed_since(running: &RunningStatus) -> String {
    let elapsed_ms = crate::model::now_ms().saturating_sub(running.started_at_ms);
    format_seconds(elapsed_ms as f64 / 1000.0)
}

fn local_area(frame: &Frame<'_>) -> Rect {
    Rect::new(0, 0, frame.area().width, frame.area().height)
}

fn page_selection(table: &mut TableState, result_len: usize, delta: isize) {
    if result_len == 0 {
        table.selection.select(None);
        table.scroll.to_start();
        return;
    }
    let current = table.selection.selected.unwrap_or(0);
    let next = if delta.is_negative() {
        current.saturating_sub(delta.unsigned_abs())
    } else {
        current
            .saturating_add(delta as usize)
            .min(result_len.saturating_sub(1))
    };
    table.selection.select(Some(next));
}

fn status_style(status: ExperimentStatus) -> Style {
    match status {
        ExperimentStatus::Keep => accent_fg(),
        ExperimentStatus::Discard => dim_fg(),
        ExperimentStatus::Crash | ExperimentStatus::ChecksFailed => warning_fg(),
    }
}

fn row_style(status: ExperimentStatus) -> Style {
    match status {
        ExperimentStatus::Keep => surface_bg(),
        ExperimentStatus::Discard => surface_bg(),
        ExperimentStatus::Crash | ExperimentStatus::ChecksFailed => surface_bg(),
    }
}

fn metric_style(direction: Option<Direction>) -> Style {
    match direction.unwrap_or(Direction::Lower) {
        Direction::Lower => Style::default().fg(Color::rgb(118, 185, 0)),
        Direction::Higher => Style::default().fg(Color::rgb(90, 170, 255)),
    }
}

fn confidence_style(confidence: f64) -> Style {
    if confidence >= 2.0 {
        accent_fg()
    } else if confidence >= 1.0 {
        warning_fg()
    } else {
        dim_fg()
    }
}

fn title_style() -> Style {
    Style::default()
        .fg(Color::rgb(197, 255, 106))
        .add_modifier(Modifier::Bold)
}

fn badge_style() -> Style {
    Style::default()
        .fg(Color::rgb(11, 15, 18))
        .bg(Color::rgb(197, 255, 106))
        .add_modifier(Modifier::Bold)
}

fn text_fg() -> Style {
    Style::default().fg(Color::rgb(226, 232, 240))
}

fn dim_fg() -> Style {
    Style::default().fg(Color::rgb(131, 145, 161))
}

fn accent_fg() -> Style {
    Style::default().fg(Color::rgb(197, 255, 106))
}

fn warning_fg() -> Style {
    Style::default().fg(Color::rgb(255, 168, 76))
}

fn surface_bg() -> Style {
    Style::default()
        .fg(Color::rgb(226, 232, 240))
        .bg(Color::rgb(13, 18, 24))
}

fn overlay_bg() -> Style {
    Style::default()
        .fg(Color::rgb(226, 232, 240))
        .bg(Color::rgb(9, 13, 17))
}

fn footer_bg() -> Style {
    Style::default()
        .fg(Color::rgb(226, 232, 240))
        .bg(Color::rgb(17, 24, 32))
}

fn header_style() -> Style {
    Style::default()
        .fg(Color::rgb(197, 255, 106))
        .bg(Color::rgb(18, 25, 33))
        .add_modifier(Modifier::Bold)
}

fn selection_style() -> Style {
    Style::default().bg(Color::rgb(25, 34, 45))
}

fn focused_selection_style() -> Style {
    Style::default()
        .fg(Color::rgb(255, 255, 255))
        .bg(Color::rgb(34, 47, 62))
}

fn border_style() -> Style {
    Style::default().fg(Color::rgb(59, 73, 87))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Direction, RunningStatus};

    fn summary(active: bool) -> StatusSummary {
        StatusSummary {
            active,
            objective: Some("speed up workspace render".to_string()),
            name: Some("workspace scroll".to_string()),
            metric_name: Some("total_ms".to_string()),
            metric_unit: "ms".to_string(),
            direction: Some(Direction::Lower),
            run_count: 3,
            kept_count: 1,
            ..StatusSummary::default()
        }
    }

    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>()
    }

    #[test]
    fn surface_effects_mount_footer_and_workspace_when_active() {
        let state = SessionUiState {
            summary: summary(true),
            expanded: true,
            overlay_open: false,
            ..SessionUiState::default()
        };
        let effects = surface_effects(&state);
        assert!(effects.iter().any(|effect| matches!(
            effect,
            TuiHostEffect::MountSurface { spec }
            if spec.key == FOOTER_KEY && spec.slot == TuiSurfaceSlot::Footer
        )));
        assert!(effects.iter().any(|effect| matches!(
            effect,
            TuiHostEffect::MountSurface { spec }
            if spec.key == WORKSPACE_KEY && spec.slot == TuiSurfaceSlot::Workspace
        )));
        assert!(effects.iter().any(|effect| matches!(
            effect,
            TuiHostEffect::UnmountSurface { key } if key == OVERLAY_KEY
        )));
    }

    #[test]
    fn surface_effects_unmount_everything_when_inactive() {
        let state = SessionUiState {
            summary: summary(false),
            expanded: true,
            overlay_open: true,
            ..SessionUiState::default()
        };
        let effects = surface_effects(&state);
        assert_eq!(effects.len(), 3);
        assert!(
            effects
                .iter()
                .all(|effect| matches!(effect, TuiHostEffect::UnmountSurface { .. }))
        );
    }

    #[test]
    fn page_selection_moves_and_clamps() {
        let mut table = TableState::default();
        page_selection(&mut table, 6, 3);
        assert_eq!(table.selection.selected, Some(3));
        page_selection(&mut table, 6, 99);
        assert_eq!(table.selection.selected, Some(5));
        page_selection(&mut table, 6, -2);
        assert_eq!(table.selection.selected, Some(3));
        page_selection(&mut table, 0, 1);
        assert_eq!(table.selection.selected, None);
    }

    #[test]
    fn footer_keeps_counts_visible_while_running() {
        let mut status = summary(true);
        status.running = Some(RunningStatus {
            command: "bash autoresearch.sh".to_string(),
            started_at_ms: crate::model::now_ms().saturating_sub(5_000),
        });
        let rendered = line_text(&footer_line(&status));
        assert!(rendered.contains("3 runs 1 kept"));
        assert!(rendered.contains("running"));
    }

    #[test]
    fn footer_mentions_waiting_before_first_logged_result() {
        let mut status = summary(true);
        status.run_count = 0;
        status.kept_count = 0;
        status.running = Some(RunningStatus {
            command: "bash autoresearch.sh".to_string(),
            started_at_ms: crate::model::now_ms().saturating_sub(5_000),
        });
        let rendered = line_text(&footer_line(&status));
        assert!(rendered.contains("0 runs 0 kept"));
        assert!(rendered.contains("waiting for first logged result"));
    }

    #[test]
    fn help_text_mentions_commands_and_shortcuts() {
        let help = autoresearch_help_text();
        assert!(help.contains("/autoresearch help"));
        assert!(help.contains("Ctrl+X"));
        assert!(help.contains("Ctrl+Shift+X"));
    }
}
