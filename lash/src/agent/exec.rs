use tokio::sync::mpsc;

use crate::session::Session;

use super::{AgentEvent, TokenUsage, send_event};

/// Accumulated state from executing code blocks during a single LLM turn.
pub(crate) struct ExecAccumulator {
    pub tool_calls: Vec<crate::ToolCallRecord>,
    pub images: Vec<crate::ToolImage>,
    pub combined_output: String,
    pub final_response: String,
    pub exec_error: Option<String>,
    pub had_failure: bool,
}

impl ExecAccumulator {
    pub fn new() -> Self {
        Self {
            tool_calls: Vec::new(),
            images: Vec::new(),
            combined_output: String::new(),
            final_response: String::new(),
            exec_error: None,
            had_failure: false,
        }
    }
}

#[allow(dead_code)]
fn clean_exec_error(raw: &str) -> &str {
    let mut cleaned = raw.trim();
    if let Some(rest) = cleaned.strip_prefix("Runtime error:") {
        cleaned = rest.trim();
    }
    cleaned
}

fn traceback_source_line(lines: &[&str]) -> Option<String> {
    lines.iter().enumerate().find_map(|(idx, line)| {
        if !line.starts_with("File ") {
            return None;
        }
        let next = lines.get(idx + 1)?.trim();
        if next.is_empty()
            || next.starts_with("File ")
            || next.starts_with("Traceback")
            || next.starts_with("During handling of the above exception")
        {
            return None;
        }
        Some(next.to_string())
    })
}

fn exception_summary_line(lines: &[&str]) -> Option<String> {
    for line in lines.iter().rev() {
        let line = line.trim();
        if line.is_empty()
            || line.starts_with("Traceback")
            || line.starts_with("File ")
            || line.starts_with('^')
            || line.starts_with('~')
            || line.starts_with("During handling of the above exception")
        {
            continue;
        }
        let Some((prefix, _)) = line.split_once(':') else {
            continue;
        };
        if prefix.is_empty() {
            continue;
        }
        let looks_like_exception = prefix
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
            && prefix
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_uppercase());
        if looks_like_exception {
            return Some(line.to_string());
        }
    }
    None
}

#[allow(dead_code)]
fn extract_exec_error_summary(raw: &str) -> String {
    let cleaned = clean_exec_error(raw);
    let lines: Vec<&str> = cleaned
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.is_empty() {
        return "Execution failed".to_string();
    }

    if let Some(summary) = exception_summary_line(&lines) {
        return summary;
    }

    lines.last().unwrap_or(&"Execution failed").to_string()
}

#[allow(dead_code)]
fn format_exec_error_for_output(raw: &str) -> String {
    if matches!(
        std::env::var("LASH_VERBOSE_RUNTIME_ERRORS")
            .ok()
            .as_deref()
            .map(|v| v.to_ascii_lowercase())
            .as_deref(),
        Some("1" | "true" | "yes" | "on")
    ) {
        return raw.trim().to_string();
    }
    let cleaned = clean_exec_error(raw);
    let lines: Vec<&str> = cleaned
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let summary = extract_exec_error_summary(raw);
    if let Some(source) = traceback_source_line(&lines)
        && source != summary
    {
        return format!("{summary}\nAt: {source}");
    }
    summary
}

/// Execute a code block, emit events, and collect results into the accumulator.
#[allow(dead_code)]
pub(crate) async fn execute_and_collect(
    session: &mut Session,
    code: &str,
    acc: &mut ExecAccumulator,
    event_tx: &mpsc::Sender<AgentEvent>,
) {
    match session.run_code(code).await {
        Ok(r) => {
            for tc in &r.tool_calls {
                send_event(
                    event_tx,
                    AgentEvent::ToolCall {
                        name: tc.tool.clone(),
                        args: tc.args.clone(),
                        result: tc.result.clone(),
                        success: tc.success,
                        duration_ms: tc.duration_ms,
                    },
                )
                .await;
                if tc.tool == "agent_result"
                    && let Some(sub) = tc.result.get("_sub_agent")
                {
                    let task = sub
                        .get("task")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let usage: TokenUsage = sub
                        .get("usage")
                        .and_then(|v| serde_json::from_value(v.clone()).ok())
                        .unwrap_or_default();
                    let tc_count =
                        sub.get("tool_calls").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    let iters =
                        sub.get("iterations").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    send_event(
                        event_tx,
                        AgentEvent::SubAgentDone {
                            task,
                            usage,
                            tool_calls: tc_count,
                            iterations: iters,
                            success: tc.success,
                        },
                    )
                    .await;
                }
            }
            if !r.output.is_empty() || r.error.is_some() {
                let normalized_error = r.error.as_deref().map(format_exec_error_for_output);
                send_event(
                    event_tx,
                    AgentEvent::CodeOutput {
                        output: r.output.clone(),
                        error: normalized_error,
                    },
                )
                .await;
            }

            acc.tool_calls.extend(r.tool_calls);
            acc.images.extend(r.images);
            if !r.output.is_empty() {
                acc.combined_output.push_str(&r.output);
            }
            if !r.response.is_empty() {
                acc.final_response = r.response;
            }
            if let Some(raw_error) = r.error {
                tracing::debug!("runtime execution error (raw): {}", raw_error);
                acc.exec_error = Some(format_exec_error_for_output(&raw_error));
                acc.had_failure = true;
            }
        }
        Err(e) => {
            let raw_error = format!("{}", e);
            tracing::debug!("runtime execution error (raw): {}", raw_error);
            send_event(
                event_tx,
                AgentEvent::CodeOutput {
                    output: String::new(),
                    error: Some(format_exec_error_for_output(&raw_error)),
                },
            )
            .await;
            acc.exec_error = Some(format_exec_error_for_output(&raw_error));
            acc.had_failure = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{extract_exec_error_summary, format_exec_error_for_output};

    #[test]
    fn summarizes_python_traceback_to_exception_line() {
        let raw = "Runtime error: Traceback (most recent call last):\n  File \"repl_1.py\", line 2, in <module>\nNameError: name 'now' is not defined";
        assert_eq!(
            extract_exec_error_summary(raw),
            "NameError: name 'now' is not defined"
        );
    }

    #[test]
    fn keeps_single_line_error() {
        let raw = "ValueError: bad input";
        assert_eq!(extract_exec_error_summary(raw), "ValueError: bad input");
    }

    #[test]
    fn does_not_mistake_code_for_exception_summary() {
        let raw = "Runtime error: Traceback (most recent call last):\n  File \"<python-input-0>\", line 1, in <module>\n    print(\"ls:\", result[\"entries\"][0][\"path\"] if result[\"entries\"] else \"empty\")\nIndexError: list index out of range";
        assert_eq!(
            extract_exec_error_summary(raw),
            "IndexError: list index out of range"
        );
    }

    #[test]
    fn formats_exception_with_source_preview() {
        let raw = "Runtime error: Traceback (most recent call last):\n  File \"<python-input-0>\", line 1, in <module>\n    print(\"ls:\", result[\"entries\"][0][\"path\"] if result[\"entries\"] else \"empty\")\nIndexError: list index out of range";
        assert_eq!(
            format_exec_error_for_output(raw),
            "IndexError: list index out of range\nAt: print(\"ls:\", result[\"entries\"][0][\"path\"] if result[\"entries\"] else \"empty\")"
        );
    }
}
