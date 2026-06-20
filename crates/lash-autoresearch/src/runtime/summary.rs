//! Status-summary projection, prompt text, and tool-result helpers.

use super::*;

pub(crate) fn prompt_text(summary: &StatusSummary) -> String {
    let objective = summary
        .objective
        .as_deref()
        .unwrap_or("No objective recorded.");
    let metric_name = summary.metric_name.as_deref().unwrap_or("metric");
    let direction = summary.direction.map(Direction::as_str).unwrap_or("lower");
    let best = summary
        .best_metric
        .map(|value| {
            format!(
                "{}{}",
                crate::model::format_metric(value),
                summary.metric_unit
            )
        })
        .unwrap_or_else(|| "—".to_string());
    let confidence = summary
        .confidence
        .map(format_confidence)
        .unwrap_or_else(|| "—".to_string());
    format!(
        "Autoresearch mode is active.\nObjective: {objective}\nCurrent metric: {metric_name} ({direction} is better).\nBest observed: {best}.\nConfidence: {confidence}.\nUse `init_experiment` once per segment, `run_experiment` for measurements, and `log_experiment` after every run. Keep improvements, revert regressions, and continue autonomously until interrupted. Keep `{}` and `{}` up to date.",
        JOURNAL_FILE, MARKDOWN_FILE
    )
}

pub(crate) fn status_event(summary: &StatusSummary) -> Result<PluginRuntimeEvent, PluginError> {
    Ok(PluginRuntimeEvent::Custom {
        name: "autoresearch.status".to_string(),
        payload: serde_json::to_value(summary).map_err(|err| {
            PluginError::Session(format!("failed to encode autoresearch status: {err}"))
        })?,
    })
}

pub(crate) fn summary_state(state: &Arc<Mutex<RuntimeState>>) -> Result<SummaryState, PluginError> {
    let state = state
        .lock()
        .map_err(|_| PluginError::Session("autoresearch state poisoned".to_string()))?;
    Ok(SummaryState {
        touched: state.touched,
        mode: state.mode.clone(),
        running: state.running.clone(),
        last_run: state.last_run.clone(),
    })
}

pub(crate) fn compute_summary_from_state(
    root: &Path,
    state: SummaryState,
) -> Result<StatusSummary, PluginError> {
    let entries = load_journal(root).map_err(PluginError::Session)?;
    Ok(compute_summary(
        &state.mode,
        &entries,
        state.running,
        state.last_run,
    ))
}

pub(crate) fn full_summary_from_runtime(
    root: &Path,
    state: &Arc<Mutex<RuntimeState>>,
) -> Result<StatusSummary, PluginError> {
    compute_summary_from_state(root, summary_state(state)?)
}

pub(crate) fn session_summary_from_runtime(
    root: &Path,
    state: &Arc<Mutex<RuntimeState>>,
) -> Result<StatusSummary, PluginError> {
    let state = summary_state(state)?;
    if !state.touched && !state.mode.active {
        return Ok(StatusSummary::default());
    }
    compute_summary_from_state(root, state)
}

pub(crate) fn status_tool_result(root: &Path, state: &Arc<Mutex<RuntimeState>>) -> ToolResult {
    match session_summary_from_runtime(root, state) {
        Ok(summary) => ToolResult::ok(json!(summary)),
        Err(err) => ToolResult::err_fmt(err.to_string()),
    }
}

pub(crate) fn tool_result_output<T>(result: ToolResult) -> Result<T, PluginOperationFailure>
where
    T: serde::de::DeserializeOwned,
{
    if !result.is_success() {
        return Err(PluginOperationFailure::new(
            result.value_for_projection().to_string(),
        ));
    }
    let output = result
        .into_done_output()
        .map_err(|_| PluginOperationFailure::new("autoresearch tool returned pending output"))?;
    serde_json::from_value(output.value_for_projection())
        .map_err(|err| PluginOperationFailure::new(format!("invalid autoresearch output: {err}")))
}

pub(crate) fn autoresearch_tool_names() -> Vec<String> {
    AUTORESEARCH_TOOL_NAMES
        .iter()
        .map(|name| (*name).to_string())
        .collect()
}
