//! Command-mode handlers (start/stop/clear/export) and argument parsing.

use super::*;

pub(crate) async fn set_autoresearch_tools_enabled(
    ctx: &PluginActionContext,
    enabled: bool,
) -> Result<(), ToolResult> {
    let Some(session_id) = ctx.session_id.as_deref() else {
        return Err(ToolResult::err_fmt(
            "autoresearch commands require a session-scoped invocation",
        ));
    };
    let availability = if enabled {
        Some(lash_core::ToolAvailability::Showcased)
    } else {
        Some(lash_core::ToolAvailability::Off)
    };
    ctx.host
        .set_tools_availability(session_id, &autoresearch_tool_names(), availability)
        .await
        .map_err(|err| {
            let action = if enabled { "enable" } else { "disable" };
            ToolResult::err_fmt(format_args!("failed to {action} autoresearch tools: {err}"))
        })?;
    Ok(())
}

pub(crate) async fn start_mode_command(
    ctx: PluginActionContext,
    root: &Path,
    state: &Arc<Mutex<RuntimeState>>,
    args: Value,
) -> ToolResult {
    if let Err(result) = set_autoresearch_tools_enabled(&ctx, true).await {
        return result;
    }
    let result = start_mode(root, state, args);
    if !result.is_success() {
        let _ = set_autoresearch_tools_enabled(&ctx, false).await;
    }
    result
}

pub(crate) async fn stop_mode_command(
    ctx: PluginActionContext,
    root: &Path,
    state: &Arc<Mutex<RuntimeState>>,
) -> ToolResult {
    if let Err(result) = set_autoresearch_tools_enabled(&ctx, false).await {
        return result;
    }
    let result = stop_mode(root, state);
    if !result.is_success() {
        let _ = set_autoresearch_tools_enabled(&ctx, true).await;
    }
    result
}

pub(crate) async fn clear_mode_command(
    ctx: PluginActionContext,
    root: &Path,
    state: &Arc<Mutex<RuntimeState>>,
) -> ToolResult {
    if let Err(result) = set_autoresearch_tools_enabled(&ctx, false).await {
        return result;
    }
    let result = clear_mode(root, state);
    if !result.is_success() {
        let _ = set_autoresearch_tools_enabled(&ctx, true).await;
    }
    result
}

pub(crate) fn start_mode(root: &Path, state: &Arc<Mutex<RuntimeState>>, args: Value) -> ToolResult {
    let objective = args
        .get("objective")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let summary = {
        let mut state = match state.lock() {
            Ok(value) => value,
            Err(_) => return ToolResult::err_fmt("autoresearch state poisoned"),
        };
        state.touched = true;
        state.mode.active = true;
        if let Some(objective) = objective.clone() {
            state.mode.objective = Some(objective);
        }
        let entries = match load_journal(root) {
            Ok(value) => value,
            Err(err) => return ToolResult::err_fmt(err),
        };
        compute_summary(
            &state.mode,
            &entries,
            state.running.clone(),
            state.last_run.clone(),
        )
    };
    if let Err(err) = rewrite_markdown(root, &summary) {
        return ToolResult::err_fmt(err);
    }
    let queued_input = objective.as_ref().map(|objective| {
        format!(
            "Start autoresearch.\nObjective: {objective}\nIf there is no active experiment segment yet, initialize one. Then make one concrete change, run a measurement, log the result, keep wins, discard regressions, and continue."
        )
    });
    ToolResult::ok(json!({
        "status": summary,
        "queued_input": queued_input,
        "message": "Autoresearch mode on."
    }))
}

pub(crate) fn stop_mode(root: &Path, state: &Arc<Mutex<RuntimeState>>) -> ToolResult {
    let summary = {
        let mut state = match state.lock() {
            Ok(value) => value,
            Err(_) => return ToolResult::err_fmt("autoresearch state poisoned"),
        };
        state.mode.active = false;
        state.running = None;
        let entries = match load_journal(root) {
            Ok(value) => value,
            Err(err) => return ToolResult::err_fmt(err),
        };
        compute_summary(&state.mode, &entries, None, state.last_run.clone())
    };
    if let Err(err) = rewrite_markdown(root, &summary) {
        return ToolResult::err_fmt(err);
    }
    ToolResult::ok(json!({
        "status": summary,
        "message": "Autoresearch mode off."
    }))
}

pub(crate) fn clear_mode(root: &Path, state: &Arc<Mutex<RuntimeState>>) -> ToolResult {
    if let Err(err) = delete_session_files(root) {
        return ToolResult::err_fmt(err);
    }
    let summary = {
        let mut state = match state.lock() {
            Ok(value) => value,
            Err(_) => return ToolResult::err_fmt("autoresearch state poisoned"),
        };
        state.touched = false;
        state.mode = ModeSnapshot::default();
        state.running = None;
        state.last_run = None;
        StatusSummary::default()
    };
    ToolResult::ok(json!({
        "status": summary,
        "message": "Cleared autoresearch session files."
    }))
}

pub(crate) fn export_summary(root: &Path, state: &Arc<Mutex<RuntimeState>>) -> ToolResult {
    let summary = match full_summary_from_runtime(root, state) {
        Ok(value) => value,
        Err(err) => return ToolResult::err_fmt(err.to_string()),
    };
    match write_export_html(root, &summary) {
        Ok(path) => ToolResult::ok(json!({
            "status": summary,
            "path": path.display().to_string(),
            "message": format!("Wrote {}.", EXPORT_FILE),
        })),
        Err(err) => ToolResult::err_fmt(err),
    }
}

pub(crate) fn require_string(args: &Value, key: &str) -> Result<String, ToolResult> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| ToolResult::err_fmt(format_args!("missing required string `{key}`")))
}

pub(crate) fn require_f64(args: &Value, key: &str) -> Result<f64, ToolResult> {
    args.get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| ToolResult::err_fmt(format_args!("missing required number `{key}`")))
}

pub(crate) fn parse_direction(value: Option<&str>) -> Result<Direction, String> {
    match value
        .unwrap_or("lower")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "lower" | "" => Ok(Direction::Lower),
        "higher" => Ok(Direction::Higher),
        other => Err(format!(
            "invalid direction `{other}`; expected `lower` or `higher`"
        )),
    }
}

pub(crate) fn parse_status(value: Option<&str>) -> Result<ExperimentStatus, String> {
    match value
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "keep" => Ok(ExperimentStatus::Keep),
        "discard" => Ok(ExperimentStatus::Discard),
        "crash" => Ok(ExperimentStatus::Crash),
        "checks_failed" => Ok(ExperimentStatus::ChecksFailed),
        other => Err(format!(
            "invalid experiment status `{other}`; expected keep, discard, crash, or checks_failed"
        )),
    }
}

pub(crate) fn parse_metrics_object(value: Option<&Value>) -> Result<BTreeMap<String, f64>, String> {
    let Some(value) = value else {
        return Ok(BTreeMap::new());
    };
    let Some(object) = value.as_object() else {
        return Err("`metrics` must be a JSON object".to_string());
    };
    let mut metrics = BTreeMap::new();
    for (name, value) in object {
        let Some(number) = value.as_f64() else {
            return Err(format!("metric `{name}` must be numeric"));
        };
        metrics.insert(name.clone(), number);
    }
    Ok(metrics)
}

pub(crate) fn parse_metric_lines(output: &str) -> BTreeMap<String, f64> {
    let mut metrics = BTreeMap::new();
    for line in output.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix("METRIC ") else {
            continue;
        };
        let Some((name, value)) = rest.split_once('=') else {
            continue;
        };
        let Ok(number) = value.trim().parse::<f64>() else {
            continue;
        };
        if number.is_finite() && !name.trim().is_empty() {
            metrics.insert(name.trim().to_string(), number);
        }
    }
    metrics
}

pub(crate) fn truncate_tail(text: &str, max_lines: usize, max_bytes: usize) -> String {
    let bytes = text.as_bytes();
    let start = bytes.len().saturating_sub(max_bytes);
    let mut tail = String::from_utf8_lossy(&bytes[start..]).into_owned();
    let lines = tail.lines().collect::<Vec<_>>();
    if lines.len() > max_lines {
        tail = lines[lines.len() - max_lines..].join("\n");
    }
    tail
}
