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

/// Execute a code block, emit events, and collect results into the accumulator.
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
                if (tc.tool == "delegate_task"
                    || tc.tool == "delegate_search"
                    || tc.tool == "delegate_deep")
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
                send_event(
                    event_tx,
                    AgentEvent::CodeOutput {
                        output: r.output.clone(),
                        error: r.error.clone(),
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
            if r.error.is_some() {
                acc.exec_error = r.error;
                acc.had_failure = true;
            }
        }
        Err(e) => {
            send_event(
                event_tx,
                AgentEvent::CodeOutput {
                    output: String::new(),
                    error: Some(format!("{}", e)),
                },
            )
            .await;
            acc.exec_error = Some(format!("{}", e));
            acc.had_failure = true;
        }
    }
}
