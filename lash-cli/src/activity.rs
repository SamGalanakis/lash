use std::{
    collections::{HashMap, HashSet},
    path::{Component, Path, PathBuf},
};

use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ActivityKind {
    Exploration,
    ShellCommand,
    ShellInteraction,
    WebSearch,
    WebFetch,
    Edit,
    Delegate,
    Parallel,
    Ask,
    GenericTool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ActivityStatus {
    Completed,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExplorationOpKind {
    Read,
    Search,
    Glob,
    List,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExplorationOp {
    pub kind: ExplorationOpKind,
    pub subject: String,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ActivityExtra {
    Exploration(Vec<ExplorationOp>),
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ActivityArtifact {
    QuestionPanel(QuestionPanelArtifact),
    DiffPreview {
        title: String,
        diff: String,
    },
    PatchPreview {
        files: Vec<PatchFilePreview>,
        total_added: usize,
        total_removed: usize,
    },
    TextPreview {
        title: Option<String>,
        text: String,
    },
    SourceList {
        title: String,
        items: Vec<String>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct QuestionPanelArtifact {
    pub prompt_lines: Vec<String>,
    pub options: Vec<QuestionPanelOption>,
    pub selection_mode: Option<QuestionPanelSelectionMode>,
    pub answer: Option<String>,
    pub note: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct QuestionPanelOption {
    pub label: String,
    pub selected: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QuestionPanelSelectionMode {
    Single,
    Multi,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PatchFilePreview {
    pub path: String,
    pub from_path: Option<String>,
    pub status: String,
    pub added: usize,
    pub removed: usize,
    pub diff: String,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ActivityBlock {
    pub kind: ActivityKind,
    pub status: ActivityStatus,
    pub tool_name: String,
    pub summary: String,
    pub detail_lines: Vec<String>,
    pub duration_ms: u64,
    pub args: Value,
    pub result: Value,
    pub artifact: Option<ActivityArtifact>,
    pub children: Vec<ActivityBlock>,
    pub extra: Option<ActivityExtra>,
}

#[derive(Default, Clone, Debug)]
pub struct ActivityState {
    shell_handles: HashMap<String, String>,
    delegate_handles: HashMap<String, String>,
}

impl ActivityState {
    pub fn reset(&mut self) {
        self.shell_handles.clear();
        self.delegate_handles.clear();
    }

    pub fn blocks_for_tool_call(
        &mut self,
        name: &str,
        args: Value,
        result: Value,
        success: bool,
        duration_ms: u64,
    ) -> Vec<ActivityBlock> {
        let name = activity_tool_name(name);
        if name == "batch" {
            let blocks = self.blocks_for_batch_tool_call(&args, &result);
            if !blocks.is_empty() {
                return blocks;
            }
        }
        // Plan content block already renders the full checklist, so no activity line needed.
        if name == "update_plan" {
            return Vec::new();
        }
        vec![self.block_for_single_tool_call(name, args, result, success, duration_ms)]
    }

    fn blocks_for_batch_tool_call(&mut self, args: &Value, result: &Value) -> Vec<ActivityBlock> {
        let Some(entries) = result
            .get("results")
            .or_else(|| result.get("details"))
            .and_then(|value| value.as_array())
        else {
            return Vec::new();
        };
        let calls = args
            .get("tool_calls")
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap_or_default();

        entries
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let child_name = item
                    .get("tool")
                    .and_then(|value| value.as_str())
                    .or_else(|| {
                        calls
                            .get(index)
                            .and_then(|value| value.get("tool"))
                            .and_then(|value| value.as_str())
                    })
                    .unwrap_or("tool");
                let child_args = calls
                    .get(index)
                    .and_then(|value| value.get("parameters"))
                    .cloned()
                    .unwrap_or_else(|| Value::Object(Default::default()));
                let child_success = item
                    .get("success")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false);
                let child_duration = item
                    .get("duration_ms")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0);
                let child_result = if child_success {
                    item.get("result").cloned().unwrap_or(Value::Null)
                } else {
                    item.get("error").cloned().unwrap_or(Value::Null)
                };

                self.block_for_single_tool_call(
                    child_name,
                    child_args,
                    child_result,
                    child_success,
                    child_duration,
                )
            })
            .collect()
    }

    fn block_for_single_tool_call(
        &mut self,
        name: &str,
        args: Value,
        result: Value,
        success: bool,
        duration_ms: u64,
    ) -> ActivityBlock {
        let name = activity_tool_name(name);
        let status = if success {
            ActivityStatus::Completed
        } else {
            ActivityStatus::Failed
        };

        match name {
            "read_file" => {
                let subject = read_label(&result).unwrap_or_else(|| {
                    compact_path_display(tool_arg_str(&args, "path").unwrap_or("file"))
                });
                exploration_block(
                    name,
                    status,
                    duration_ms,
                    args,
                    result,
                    ExplorationOp {
                        kind: ExplorationOpKind::Read,
                        subject,
                    },
                )
            }
            "grep" => {
                let subject = grep_label(&args);
                exploration_block(
                    name,
                    status,
                    duration_ms,
                    args,
                    result,
                    ExplorationOp {
                        kind: ExplorationOpKind::Search,
                        subject,
                    },
                )
            }
            "glob" => {
                let subject = glob_label(&args);
                exploration_block(
                    name,
                    status,
                    duration_ms,
                    args,
                    result,
                    ExplorationOp {
                        kind: ExplorationOpKind::Glob,
                        subject,
                    },
                )
            }
            "ls" => {
                let subject = compact_path_display(tool_arg_str(&args, "path").unwrap_or("."));
                exploration_block(
                    name,
                    status,
                    duration_ms,
                    args,
                    result,
                    ExplorationOp {
                        kind: ExplorationOpKind::List,
                        subject,
                    },
                )
            }
            "exec_command" => {
                let command = tool_arg_str(&args, "cmd")
                    .map(str::to_string)
                    .unwrap_or_else(|| "command".to_string());
                let running_handle =
                    tool_result_shell_id(&result).filter(|_| shell_result_running(&result));
                if let Some(handle_id) = running_handle.as_ref() {
                    self.shell_handles
                        .insert(handle_id.clone(), command.clone());
                }
                let exit_code = shell_result_exit_code(&result);
                let status = if !success || exit_code.is_some_and(|value| value != 0) {
                    ActivityStatus::Failed
                } else {
                    ActivityStatus::Completed
                };
                let mut detail_lines = Vec::new();
                if status == ActivityStatus::Failed {
                    if let Some(exit_code) = exit_code {
                        detail_lines.push(format!("Exited with {}", exit_code));
                    }
                    if let Some(workdir) = tool_arg_str(&args, "workdir") {
                        detail_lines.push(format!("In {}", workdir));
                    }
                } else if let Some(ref handle_id) = running_handle {
                    detail_lines.push(format!("Handle {}", handle_id));
                }
                let artifact = shell_output_artifact(&result);
                let summary = if running_handle.is_some() {
                    shell_start_summary(&command, status)
                } else if status == ActivityStatus::Failed {
                    format!("failed {}", inline_text(&command))
                } else {
                    inline_text(&command)
                };
                ActivityBlock {
                    kind: ActivityKind::ShellCommand,
                    status,
                    tool_name: name.to_string(),
                    summary,
                    detail_lines,
                    duration_ms,
                    args,
                    result,
                    artifact,
                    children: Vec::new(),
                    extra: None,
                }
            }
            "write_stdin" => {
                let handle_id = tool_arg_shell_id(&args).unwrap_or_default();
                let command = self
                    .shell_handles
                    .get(&handle_id)
                    .cloned()
                    .unwrap_or_else(|| format!("shell {}", handle_id));
                let exit_code = shell_result_exit_code(&result);
                let running = shell_result_running(&result);
                if exit_code.is_some() {
                    self.shell_handles.remove(&handle_id);
                } else if running && !handle_id.is_empty() {
                    self.shell_handles
                        .entry(handle_id.clone())
                        .or_insert_with(|| command.clone());
                }
                let status = if !success || exit_code.is_some_and(|value| value != 0) {
                    ActivityStatus::Failed
                } else {
                    ActivityStatus::Completed
                };
                let chars = tool_arg_str(&args, "chars").unwrap_or("");
                let mut detail_lines = Vec::new();
                if !chars.is_empty() {
                    detail_lines.push(format!("Input {}", inline_text(chars)));
                }
                if let Some(exit_code) = exit_code {
                    detail_lines.push(format!("Exited with {}", exit_code));
                } else if running && !handle_id.is_empty() {
                    detail_lines.push(format!("Handle {}", handle_id));
                }
                let artifact = shell_output_artifact(&result);
                let summary = shell_write_summary(&command, chars, &result, status);
                ActivityBlock {
                    kind: ActivityKind::ShellInteraction,
                    status,
                    tool_name: name.to_string(),
                    summary,
                    detail_lines,
                    duration_ms,
                    args,
                    result,
                    artifact,
                    children: Vec::new(),
                    extra: None,
                }
            }
            "search_web" => {
                let query = tool_arg_str(&args, "query")
                    .unwrap_or("search web")
                    .to_string();
                let items = web_sources(&result);
                let artifact = (!items.is_empty()).then(|| ActivityArtifact::SourceList {
                    title: "Sources".to_string(),
                    items,
                });
                ActivityBlock {
                    kind: ActivityKind::WebSearch,
                    status,
                    tool_name: name.to_string(),
                    summary: web_search_summary(&query),
                    detail_lines: web_search_detail_lines(&result),
                    duration_ms,
                    args,
                    result,
                    artifact,
                    children: Vec::new(),
                    extra: None,
                }
            }
            "fetch_url" => {
                let url = tool_arg_str(&args, "url").unwrap_or("url").to_string();
                let artifact = text_preview_artifact(Some("Fetched content"), &result);
                ActivityBlock {
                    kind: ActivityKind::WebFetch,
                    status,
                    tool_name: name.to_string(),
                    summary: format!("fetch {}", display_url(&url)),
                    detail_lines: Vec::new(),
                    duration_ms,
                    args,
                    result,
                    artifact,
                    children: Vec::new(),
                    extra: None,
                }
            }
            "apply_patch" => {
                let artifact = patch_preview_artifact(&result);
                let summary = patch_summary(&result)
                    .or_else(|| {
                        result
                            .get("summary")
                            .and_then(|value| value.as_str())
                            .map(str::to_string)
                    })
                    .unwrap_or_else(|| semantic_tool_summary(name, &args));
                ActivityBlock {
                    kind: ActivityKind::Edit,
                    status,
                    tool_name: name.to_string(),
                    summary,
                    detail_lines: Vec::new(),
                    duration_ms,
                    args,
                    result,
                    artifact,
                    children: Vec::new(),
                    extra: None,
                }
            }
            "agent_call" => {
                let task = tool_arg_str(&args, "task")
                    .or_else(|| tool_arg_str(&args, "prompt"))
                    .unwrap_or("delegate task")
                    .to_string();
                if success && let Some(id) = tool_result_handle_id(&result) {
                    self.delegate_handles.insert(id.to_string(), task.clone());
                }
                let mut detail_lines = Vec::new();
                if success {
                    let model = result
                        .get("model")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    let variant = result
                        .get("model_variant")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    if !model.is_empty() {
                        let label = if variant.is_empty() {
                            model.to_string()
                        } else {
                            format!("{model} ({variant})")
                        };
                        detail_lines.push(label);
                    }
                }
                ActivityBlock {
                    kind: ActivityKind::Delegate,
                    status,
                    tool_name: name.to_string(),
                    summary: format!("delegate · {}", inline_text(&task)),
                    detail_lines,
                    duration_ms,
                    args,
                    result,
                    artifact: None,
                    children: Vec::new(),
                    extra: None,
                }
            }
            "agent_result" => {
                let handle_id = tool_arg_str(&args, "id").unwrap_or_default().to_string();
                let meta = result.get("session");
                let child_status = result
                    .get("status")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default();
                let task = meta
                    .and_then(|value| value.get("task"))
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
                    .or_else(|| self.delegate_handles.remove(&handle_id))
                    .unwrap_or_else(|| handle_id.clone());

                let delegate_status = delegate_activity_status(success, child_status);
                let delegate_summary = delegate_result_summary(child_status, &task);
                let mut detail_lines = Vec::new();
                if let Some(error) = result.get("error").and_then(|value| value.as_str())
                    && !error.trim().is_empty()
                {
                    detail_lines.push(format!("Error {}", inline_text(error)));
                }

                if let Some(meta) = meta {
                    // Model info line.
                    let model = meta
                        .get("model")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    let variant = meta
                        .get("model_variant")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    let tool_calls = meta.get("tool_calls").and_then(|v| v.as_u64());
                    let iterations = meta.get("iterations").and_then(|v| v.as_u64());

                    // Summary stats line.
                    let mut parts = Vec::new();
                    if !model.is_empty() {
                        let label = if variant.is_empty() {
                            model.to_string()
                        } else {
                            format!("{model} ({variant})")
                        };
                        parts.push(label);
                    }
                    if let Some(iters) = iterations {
                        parts.push(format!(
                            "{} iteration{}",
                            iters,
                            if iters == 1 { "" } else { "s" }
                        ));
                    }
                    if let Some(tc) = tool_calls {
                        parts.push(format!(
                            "{} tool call{}",
                            tc,
                            if tc == 1 { "" } else { "s" }
                        ));
                    }
                    if !parts.is_empty() {
                        detail_lines.push(parts.join(" · "));
                    }
                    if let Some(token_line) = delegate_token_usage_line(meta) {
                        detail_lines.push(token_line);
                    }
                }

                let artifact = result
                    .get("result")
                    .and_then(|value| value.as_str())
                    .filter(|text| !text.trim().is_empty())
                    .map(|text| ActivityArtifact::TextPreview {
                        title: Some("Delegate result".to_string()),
                        text: text.to_string(),
                    });
                ActivityBlock {
                    kind: ActivityKind::Delegate,
                    status: delegate_status,
                    tool_name: name.to_string(),
                    summary: delegate_summary,
                    detail_lines,
                    duration_ms,
                    args,
                    result,
                    artifact,
                    children: Vec::new(),
                    extra: None,
                }
            }
            "agent_kill" => {
                let handle_id = tool_arg_str(&args, "id").unwrap_or_default().to_string();
                let task = self
                    .delegate_handles
                    .remove(&handle_id)
                    .unwrap_or(handle_id);
                ActivityBlock {
                    kind: ActivityKind::Delegate,
                    status,
                    tool_name: name.to_string(),
                    summary: format!("delegate stopped · {}", inline_text(&task)),
                    detail_lines: Vec::new(),
                    duration_ms,
                    args,
                    result,
                    artifact: None,
                    children: Vec::new(),
                    extra: None,
                }
            }
            "ask" => {
                let detail_lines = ask_detail_lines(&args, &result);
                let artifact = inline_question_panel_artifact(&args, &result);
                ActivityBlock {
                    kind: ActivityKind::Ask,
                    status,
                    tool_name: name.to_string(),
                    summary: ask_summary(&args),
                    detail_lines,
                    duration_ms,
                    args,
                    artifact,
                    children: Vec::new(),
                    extra: None,
                    result,
                }
            }
            // update_plan is filtered out in blocks_for_tool_call; this branch
            // is unreachable but kept as a guard.
            "update_plan" => ActivityBlock {
                kind: ActivityKind::GenericTool,
                status,
                tool_name: name.to_string(),
                summary: "updated plan".to_string(),
                detail_lines: Vec::new(),
                duration_ms,
                args,
                result,
                artifact: None,
                children: Vec::new(),
                extra: None,
            },
            "search_tools" => ActivityBlock {
                kind: ActivityKind::GenericTool,
                status,
                tool_name: name.to_string(),
                summary: tool_search_summary(&args),
                detail_lines: tool_search_detail_lines(&result),
                duration_ms,
                args,
                artifact: None,
                children: Vec::new(),
                extra: None,
                result,
            },
            _ => ActivityBlock {
                kind: ActivityKind::GenericTool,
                status,
                tool_name: name.to_string(),
                summary: semantic_tool_summary(name, &args),
                detail_lines: Vec::new(),
                duration_ms,
                args,
                artifact: { text_preview_artifact(None, &result) },
                children: Vec::new(),
                extra: None,
                result,
            },
        }
    }
}

fn activity_tool_name(name: &str) -> &str {
    name.strip_prefix("functions.")
        .or_else(|| name.strip_prefix("web."))
        .unwrap_or(name)
}

pub fn merge_exploration_activity(target: &mut ActivityBlock, mut incoming: ActivityBlock) -> bool {
    let Some(ActivityExtra::Exploration(target_ops)) = target.extra.as_mut() else {
        return false;
    };
    let Some(ActivityExtra::Exploration(incoming_ops)) = incoming.extra.take() else {
        return false;
    };
    target_ops.extend(incoming_ops);
    target.duration_ms += incoming.duration_ms;
    rebuild_exploration_summary(target);
    true
}

pub fn merge_edit_activity(target: &mut ActivityBlock, incoming: ActivityBlock) -> bool {
    let Some(ActivityArtifact::PatchPreview {
        files,
        total_added,
        total_removed,
    }) = target.artifact.as_mut()
    else {
        return false;
    };
    let Some(ActivityArtifact::PatchPreview {
        files: incoming_files,
        total_added: incoming_added,
        total_removed: incoming_removed,
    }) = incoming.artifact.clone()
    else {
        return false;
    };

    files.extend(incoming_files);
    *total_added += incoming_added;
    *total_removed += incoming_removed;
    target.duration_ms += incoming.duration_ms;
    target.summary = patch_summary_from_preview(files, *total_added, *total_removed);
    true
}

fn rebuild_exploration_summary(block: &mut ActivityBlock) {
    let Some(ActivityExtra::Exploration(ops)) = block.extra.as_ref() else {
        return;
    };
    block.summary = exploration_summary(ops);
    block.detail_lines = ops.iter().map(exploration_step_line).collect();
}

fn semantic_tool_summary(name: &str, args: &Value) -> String {
    match name {
        "read_file" => tool_arg_str(args, "path")
            .map(|path| {
                let mut label = format!("read {}", compact_path_display(path));
                if let Some(offset) = args.get("offset").and_then(|value| value.as_u64())
                    && offset > 1
                {
                    label.push_str(&format!(" @{}", offset));
                }
                label
            })
            .unwrap_or_else(|| "read file".to_string()),
        "apply_patch" => "apply patch".to_string(),
        "grep" => grep_label(args),
        "glob" => glob_label(args),
        "ls" => format!(
            "list {}",
            compact_path_display(tool_arg_str(args, "path").unwrap_or("."))
        ),
        "exec_command" => tool_arg_str(args, "cmd")
            .map(inline_text)
            .unwrap_or_else(|| "command".to_string()),
        "write_stdin" => tool_arg_str(args, "chars")
            .map(|chars| {
                if chars.is_empty() {
                    "poll command output".to_string()
                } else {
                    "write to command".to_string()
                }
            })
            .unwrap_or_else(|| "write to command".to_string()),
        "fetch_url" => tool_arg_str(args, "url")
            .map(|url| format!("fetch {}", display_url(url)))
            .unwrap_or_else(|| "fetch url".to_string()),
        "search_web" => tool_arg_str(args, "query")
            .map(|query| format!("web \"{}\"", inline_text(query)))
            .unwrap_or_else(|| "search web".to_string()),
        "agent_call" => tool_arg_str(args, "task")
            .or_else(|| tool_arg_str(args, "prompt"))
            .map(|task| format!("delegate · {}", inline_text(task)))
            .unwrap_or_else(|| "delegate task".to_string()),
        "agent_result" => "delegate done".to_string(),
        "agent_kill" => "delegate stopped".to_string(),
        _ => name.replace('_', " "),
    }
}

fn delegate_activity_status(success: bool, child_status: &str) -> ActivityStatus {
    if !success || matches!(child_status, "failed" | "interrupted") {
        ActivityStatus::Failed
    } else {
        ActivityStatus::Completed
    }
}

fn delegate_result_summary(child_status: &str, task: &str) -> String {
    let label = match child_status {
        "interrupted" => "delegate stopped",
        "failed" => "delegate failed",
        _ => "delegate done",
    };
    format!("{label} · {}", inline_text(task))
}

fn delegate_token_usage_line(meta: &Value) -> Option<String> {
    let usage = meta.get("token_usage")?;
    let total = usage.get("total_tokens").and_then(|value| value.as_u64());
    let input = usage.get("input_tokens").and_then(|value| value.as_u64());
    let output = usage.get("output_tokens").and_then(|value| value.as_u64());
    let reasoning = usage
        .get("reasoning_tokens")
        .and_then(|value| value.as_u64())
        .filter(|value| *value > 0);
    let cached = usage
        .get("cached_input_tokens")
        .and_then(|value| value.as_u64())
        .filter(|value| *value > 0);

    let mut parts = Vec::new();
    if let Some(total) = total {
        parts.push(format!("{total} total tokens"));
    }
    if let Some(input) = input {
        parts.push(format!("{input} in"));
    }
    if let Some(output) = output {
        parts.push(format!("{output} out"));
    }
    if let Some(reasoning) = reasoning {
        parts.push(format!("{reasoning} reasoning"));
    }
    if let Some(cached) = cached {
        parts.push(format!("{cached} cached"));
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" · "))
    }
}

fn exploration_block(
    name: &str,
    status: ActivityStatus,
    duration_ms: u64,
    args: Value,
    result: Value,
    op: ExplorationOp,
) -> ActivityBlock {
    let mut block = ActivityBlock {
        kind: ActivityKind::Exploration,
        status,
        tool_name: name.to_string(),
        summary: String::new(),
        detail_lines: Vec::new(),
        duration_ms,
        args,
        result,
        artifact: None,
        children: Vec::new(),
        extra: Some(ActivityExtra::Exploration(vec![op])),
    };
    rebuild_exploration_summary(&mut block);
    block
}

fn exploration_summary(ops: &[ExplorationOp]) -> String {
    let count = ops.len();
    if count == 1 {
        "EXPLORE · 1 step".to_string()
    } else {
        format!("EXPLORE · {count} steps")
    }
}

fn exploration_step_line(op: &ExplorationOp) -> String {
    match op.kind {
        ExplorationOpKind::Read => format!("Read {}", op.subject),
        ExplorationOpKind::Search => format!("Search {}", op.subject),
        ExplorationOpKind::Glob => format!("Glob {}", op.subject),
        ExplorationOpKind::List => format!("List {}", op.subject),
    }
}

fn grep_label(args: &Value) -> String {
    let pattern = tool_arg_str(args, "pattern").unwrap_or("pattern");
    if let Some(path) = tool_arg_str(args, "path") {
        format!("{:?} in {}", pattern, compact_path_display(path))
    } else {
        format!("{:?}", pattern)
    }
}

fn glob_label(args: &Value) -> String {
    let pattern = tool_arg_str(args, "pattern").unwrap_or("*");
    if let Some(path) = tool_arg_str(args, "path") {
        format!("{} in {}", pattern, compact_path_display(path))
    } else {
        pattern.to_string()
    }
}

fn read_label(result: &Value) -> Option<String> {
    let text = result.as_str()?;
    let first_line = text.lines().next()?.trim();
    if first_line.is_empty() {
        return None;
    }
    if let Some(rest) = first_line.strip_prefix("==> ")
        && let Some(path) = rest.strip_suffix(" <==")
    {
        return Some(compact_path_display(path));
    }
    if let Some(path) = first_line.split_whitespace().next()
        && path.contains('/')
    {
        return Some(compact_path_display(path));
    }
    None
}

pub(crate) fn compact_path_display(path: &str) -> String {
    let path = Path::new(path);
    if !path.is_absolute() {
        return path.to_string_lossy().into_owned();
    }

    if let Ok(cwd) = std::env::current_dir()
        && let Ok(relative) = path.strip_prefix(&cwd)
    {
        let label = relative.to_string_lossy();
        return if label.is_empty() {
            ".".to_string()
        } else {
            label.into_owned()
        };
    }

    if let Some(home) = user_home_dir()
        && let Ok(relative) = path.strip_prefix(home)
    {
        return compact_home_relative_path(relative);
    }

    compact_absolute_path(path)
}

fn user_home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn compact_home_relative_path(path: &Path) -> String {
    let components = display_path_components(path);
    if components.is_empty() {
        return "~".to_string();
    }
    if components.len() <= 3 {
        return format!("~/{}", components.join("/"));
    }
    format!("~/…/{}", tail_components(&components, 3))
}

fn compact_absolute_path(path: &Path) -> String {
    let components = display_path_components(path);
    if components.is_empty() {
        return "/".to_string();
    }
    if components.len() <= 3 {
        return format!("/{}", components.join("/"));
    }
    format!("…/{}", tail_components(&components, 3))
}

fn display_path_components(path: &Path) -> Vec<String> {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(value) => Some(value.to_string_lossy().into_owned()),
            Component::CurDir => Some(".".to_string()),
            Component::ParentDir => Some("..".to_string()),
            Component::RootDir | Component::Prefix(_) => None,
        })
        .collect()
}

fn tail_components(components: &[String], count: usize) -> String {
    components
        .iter()
        .skip(components.len().saturating_sub(count))
        .cloned()
        .collect::<Vec<_>>()
        .join("/")
}

fn web_sources(result: &Value) -> Vec<String> {
    result
        .get("results")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("url").and_then(|value| value.as_str()))
                .take(5)
                .map(|url| url.to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn web_search_summary(query: &str) -> String {
    format!("searched web for {:?}", inline_text(query))
}

fn web_search_detail_lines(result: &Value) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(answer) = result.get("answer").and_then(|value| value.as_str())
        && !answer.trim().is_empty()
    {
        lines.push(format!("Answer {}", inline_snippet(answer, 72)));
    }
    if let Some(results) = result.get("results").and_then(|value| value.as_array()) {
        lines.extend(results.iter().take(3).filter_map(|item| {
            let title = item.get("title").and_then(|value| value.as_str())?;
            let url = item
                .get("url")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            if url.is_empty() {
                Some(inline_snippet(title, 72))
            } else {
                Some(format!(
                    "{} · {}",
                    inline_snippet(title, 48),
                    display_url(url)
                ))
            }
        }));
    }
    lines
}

fn text_preview_artifact(title: Option<&str>, result: &Value) -> Option<ActivityArtifact> {
    let text = if let Some(text) = result.as_str() {
        text.to_string()
    } else if let Some(text) = result.get("answer").and_then(|value| value.as_str()) {
        text.to_string()
    } else {
        return None;
    };

    if text.trim().is_empty() {
        return None;
    }

    Some(ActivityArtifact::TextPreview {
        title: title.map(str::to_string),
        text,
    })
}

fn tool_result_handle_id(result: &Value) -> Option<&str> {
    result
        .get("id")
        .and_then(|value| value.as_str())
        .or_else(|| result.get("session_id").and_then(|value| value.as_str()))
}

fn tool_result_shell_id(result: &Value) -> Option<String> {
    result
        .get("id")
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .or_else(|| {
            result
                .get("session_id")
                .and_then(|value| value.as_i64())
                .map(|value| value.to_string())
        })
        .or_else(|| {
            result
                .get("session_id")
                .and_then(|value| value.as_u64())
                .map(|value| value.to_string())
        })
        .or_else(|| {
            result
                .get("session_id")
                .and_then(|value| value.as_str())
                .map(str::to_string)
        })
}

fn tool_arg_shell_id(args: &Value) -> Option<String> {
    args.get("id")
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .or_else(|| {
            args.get("session_id")
                .and_then(|value| value.as_i64())
                .map(|value| value.to_string())
        })
        .or_else(|| {
            args.get("session_id")
                .and_then(|value| value.as_u64())
                .map(|value| value.to_string())
        })
        .or_else(|| {
            args.get("session_id")
                .and_then(|value| value.as_str())
                .map(str::to_string)
        })
}

fn shell_result_output(result: &Value) -> Option<String> {
    result
        .get("output")
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

fn shell_output_artifact(result: &Value) -> Option<ActivityArtifact> {
    shell_result_output(result).and_then(|text| {
        if text.trim().is_empty() {
            None
        } else {
            Some(ActivityArtifact::TextPreview {
                title: Some("Shell output".to_string()),
                text,
            })
        }
    })
}

fn ask_summary(args: &Value) -> String {
    let _ = args;
    "Question".to_string()
}

fn ask_detail_lines(args: &Value, _result: &Value) -> Vec<String> {
    let mut lines = Vec::new();

    if let Some(question) = tool_arg_str(args, "question") {
        let mut question_lines = question
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(inline_text);
        if let Some(first_line) = question_lines.next() {
            lines.push(first_line);
            lines.extend(question_lines);
        }
    }

    for (idx, option) in tool_arg_list(args, "options").into_iter().enumerate() {
        lines.push(format!("{}. {}", idx + 1, option));
    }

    lines
}

fn prompt_lines(question: &str) -> Vec<String> {
    question
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect()
}

fn inline_question_panel_artifact(args: &Value, result: &Value) -> Option<ActivityArtifact> {
    let question = tool_arg_str(args, "question")?;
    let mut options = tool_arg_list(args, "options")
        .into_iter()
        .map(|label| QuestionPanelOption {
            label,
            selected: false,
        })
        .collect::<Vec<_>>();
    let mut selection_mode = (!options.is_empty()).then_some(QuestionPanelSelectionMode::Single);
    let note = result
        .get("note")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let mut answer = None;

    match result.get("kind").and_then(|value| value.as_str()) {
        Some("single") => {
            selection_mode = Some(QuestionPanelSelectionMode::Single);
            if let Some(selection) = result.get("selection").and_then(|value| value.as_str()) {
                if let Some(option) = options.iter_mut().find(|option| option.label == selection) {
                    option.selected = true;
                } else if !selection.trim().is_empty() {
                    answer = Some(selection.trim().to_string());
                }
            }
        }
        Some("multi") => {
            selection_mode = Some(QuestionPanelSelectionMode::Multi);
            let mut unmatched = Vec::new();
            for selection in result
                .get("selections")
                .and_then(|value| value.as_array())
                .into_iter()
                .flatten()
                .filter_map(|value| value.as_str())
            {
                if let Some(option) = options.iter_mut().find(|option| option.label == selection) {
                    option.selected = true;
                } else if !selection.trim().is_empty() {
                    unmatched.push(selection.trim().to_string());
                }
            }
            if !unmatched.is_empty() {
                answer = Some(unmatched.join(", "));
            }
        }
        Some("text") => {
            answer = result
                .get("text")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string);
        }
        _ => {}
    }

    Some(ActivityArtifact::QuestionPanel(QuestionPanelArtifact {
        prompt_lines: prompt_lines(question),
        options,
        selection_mode,
        answer,
        note,
    }))
}

fn tool_search_summary(args: &Value) -> String {
    tool_arg_str(args, "query")
        .map(|query| format!("searched tools for {:?}", inline_text(query)))
        .unwrap_or_else(|| "browsed tools".to_string())
}

fn tool_search_detail_lines(result: &Value) -> Vec<String> {
    named_description_detail_lines(result, 4)
}

fn shell_start_summary(command: &str, status: ActivityStatus) -> String {
    if status == ActivityStatus::Failed {
        format!("failed {}", inline_text(command))
    } else {
        format!("started {}", inline_text(command))
    }
}

fn shell_write_summary(
    command: &str,
    input: &str,
    result: &Value,
    status: ActivityStatus,
) -> String {
    if status == ActivityStatus::Failed {
        return format!("failed {}", inline_text(command));
    }
    if input.trim().is_empty() {
        if shell_result_output(result).is_some_and(|text| !text.trim().is_empty()) {
            return format!("read {}", inline_text(command));
        }
        if shell_result_running(result) {
            return format!("polled {}", inline_text(command));
        }
        return format!("finished {}", inline_text(command));
    }
    format!(
        "sent {} → {}",
        shell_input_preview(input),
        inline_text(command)
    )
}

fn shell_input_preview(input: &str) -> String {
    const MAX_PREVIEW_CHARS: usize = 24;
    let compact = inline_text(input);
    let preview: String = compact.chars().take(MAX_PREVIEW_CHARS).collect();
    if compact.chars().count() > MAX_PREVIEW_CHARS {
        format!("{preview}...")
    } else {
        preview
    }
}

fn shell_result_running(result: &Value) -> bool {
    result
        .get("running")
        .and_then(|value| value.as_bool())
        .unwrap_or_else(|| {
            result
                .get("session_id")
                .is_some_and(|value| !value.is_null())
        })
}

fn shell_result_exit_code(result: &Value) -> Option<i64> {
    result.get("exit_code").and_then(|value| value.as_i64())
}

fn named_description_detail_lines(result: &Value, limit: usize) -> Vec<String> {
    result
        .as_array()
        .map(|items| {
            items
                .iter()
                .take(limit)
                .filter_map(|item| {
                    let name = item.get("name").and_then(|value| value.as_str())?;
                    let description = item
                        .get("description")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default();
                    if description.trim().is_empty() {
                        Some(name.to_string())
                    } else {
                        Some(format!("{name}: {}", inline_snippet(description, 72)))
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

fn tool_arg_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key)
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
}

fn tool_arg_list(args: &Value, key: &str) -> Vec<String> {
    args.get(key)
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|value| match value {
                    Value::String(text) if !text.is_empty() => Some(inline_text(text)),
                    Value::Bool(_) | Value::Number(_) => Some(value.to_string()),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

fn patch_preview_artifact(result: &Value) -> Option<ActivityArtifact> {
    let files = result
        .get("files")
        .and_then(|value| value.as_array())
        .map(|files| {
            files
                .iter()
                .filter_map(|file| {
                    let path = file.get("path").and_then(|value| value.as_str())?;
                    let status = file
                        .get("status")
                        .and_then(|value| value.as_str())
                        .unwrap_or("modified")
                        .to_string();
                    let from_path = file
                        .get("from_path")
                        .and_then(|value| value.as_str())
                        .map(str::to_string);
                    let diff = file
                        .get("diff")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let (added, removed) = patch_file_counts(file, &diff);
                    Some(PatchFilePreview {
                        path: path.to_string(),
                        from_path,
                        status,
                        added,
                        removed,
                        diff,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    if files.is_empty() {
        return result
            .get("diff")
            .and_then(|value| value.as_str())
            .filter(|diff| !diff.trim().is_empty())
            .map(|diff| ActivityArtifact::DiffPreview {
                title: "Diff".to_string(),
                diff: diff.to_string(),
            });
    }

    let total_added = result
        .get("added")
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .unwrap_or_else(|| files.iter().map(|file| file.added).sum());
    let total_removed = result
        .get("removed")
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .unwrap_or_else(|| files.iter().map(|file| file.removed).sum());

    Some(ActivityArtifact::PatchPreview {
        files,
        total_added,
        total_removed,
    })
}

fn patch_summary(result: &Value) -> Option<String> {
    let ActivityArtifact::PatchPreview {
        files,
        total_added,
        total_removed,
    } = patch_preview_artifact(result)?
    else {
        return None;
    };

    Some(patch_summary_from_preview(
        &files,
        total_added,
        total_removed,
    ))
}

fn patch_summary_from_preview(
    files: &[PatchFilePreview],
    total_added: usize,
    total_removed: usize,
) -> String {
    if files.len() == 1 {
        let file = &files[0];
        return format!(
            "{} {} {}",
            patch_status_title(&file.status),
            patch_file_subject(file),
            patch_count_suffix(file.added, file.removed)
        );
    }

    let file_count = patch_unique_file_count(files);
    let noun = if file_count == 1 { "file" } else { "files" };
    format!(
        "{} {} {} {}",
        patch_group_title(files),
        file_count,
        noun,
        patch_count_suffix(total_added, total_removed)
    )
}

fn patch_group_title(files: &[PatchFilePreview]) -> &'static str {
    let Some(first) = files.first() else {
        return "Edited";
    };
    if files.iter().all(|file| file.status == first.status) {
        patch_status_title(&first.status)
    } else {
        "Edited"
    }
}

pub(crate) fn patch_status_title(status: &str) -> &'static str {
    match status {
        "added" => "Added",
        "deleted" => "Deleted",
        "moved" => "Moved",
        _ => "Edited",
    }
}

fn patch_unique_file_count(files: &[PatchFilePreview]) -> usize {
    let mut unique = HashSet::new();
    for file in files {
        unique.insert(patch_file_subject(file));
    }
    unique.len()
}

pub(crate) fn patch_file_subject(file: &PatchFilePreview) -> String {
    let path = compact_path_display(&file.path);
    match &file.from_path {
        Some(from_path) => format!("{} → {path}", compact_path_display(from_path)),
        None => path,
    }
}

fn patch_count_suffix(added: usize, removed: usize) -> String {
    format!("(+{} -{})", added, removed)
}

fn patch_file_counts(file: &Value, diff: &str) -> (usize, usize) {
    let added = file
        .get("added")
        .and_then(|value| value.as_u64())
        .map(|value| value as usize);
    let removed = file
        .get("removed")
        .and_then(|value| value.as_u64())
        .map(|value| value as usize);
    match (added, removed) {
        (Some(added), Some(removed)) => (added, removed),
        _ => count_diff_delta(diff),
    }
}

fn count_diff_delta(diff: &str) -> (usize, usize) {
    let mut added = 0usize;
    let mut removed = 0usize;
    for line in diff.lines() {
        if line.starts_with("+++ ") || line.starts_with("--- ") || line.starts_with("@@") {
            continue;
        }
        if line.starts_with('+') {
            added += 1;
        } else if line.starts_with('-') {
            removed += 1;
        }
    }
    (added, removed)
}

fn inline_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn inline_snippet(text: &str, max_chars: usize) -> String {
    let compact = inline_text(text);
    let snippet: String = compact.chars().take(max_chars).collect();
    if compact.chars().count() > max_chars {
        format!("{snippet}...")
    } else {
        snippet
    }
}

fn display_url(url: &str) -> String {
    url.trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_end_matches('/')
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn apply_patch_summary_prefers_semantic_single_file_copy() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
            "apply_patch",
            json!({}),
            json!({
                "summary": "Applied patch to 1 file",
                "added": 3,
                "removed": 1,
                "files": [{
                    "path": "lash-cli/src/ui.rs",
                    "status": "modified",
                    "added": 3,
                    "removed": 1,
                    "diff": "--- a/lash-cli/src/ui.rs\n+++ b/lash-cli/src/ui.rs\n@@\n-old\n+new"
                }]
            }),
            true,
            18,
        );

        assert_eq!(blocks[0].summary, "Edited lash-cli/src/ui.rs (+3 -1)");
        assert!(matches!(
            blocks[0].artifact,
            Some(ActivityArtifact::PatchPreview {
                total_added: 3,
                total_removed: 1,
                ..
            })
        ));
    }

    #[test]
    fn apply_patch_summary_shows_move_arrow() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
            "apply_patch",
            json!({}),
            json!({
                "summary": "Applied patch to 1 file",
                "added": 2,
                "removed": 2,
                "files": [{
                    "path": "new.rs",
                    "from_path": "old.rs",
                    "status": "moved",
                    "added": 2,
                    "removed": 2,
                    "diff": "--- a/new.rs\n+++ b/new.rs\n@@\n-old\n+new"
                }]
            }),
            true,
            11,
        );

        assert_eq!(blocks[0].summary, "Moved old.rs → new.rs (+2 -2)");
    }

    #[test]
    fn exec_command_success_uses_plain_command_summary() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
            "exec_command",
            json!({
                "cmd": "date '+%Y-%m-%d %H:%M:%S %Z'",
                "workdir": "/home/sam/code/lash"
            }),
            json!({
                "output": "2026-03-12 17:11:12 CET",
                "exit_code": 0
            }),
            true,
            13,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].summary, "date '+%Y-%m-%d %H:%M:%S %Z'");
    }

    #[test]
    fn exec_command_running_uses_session_id_handle() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
            "exec_command",
            json!({
                "cmd": "python3 -q",
            }),
            json!({
                "output": ">>> ",
                "session_id": 7,
                "wall_time_seconds": 0.01
            }),
            true,
            13,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].summary, "started python3 -q");
        assert!(blocks[0].detail_lines.iter().any(|line| line == "Handle 7"));
    }

    #[test]
    fn batch_expands_into_child_tool_blocks() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
            "batch",
            json!({
                "tool_calls": [
                    {"tool": "read_file", "parameters": {"path": "a.rs"}},
                    {"tool": "grep", "parameters": {"pattern": "foo", "path": "."}}
                ]
            }),
            json!({
                "summary": "Executed 1/2 tools successfully. 1 failed.",
                "results": [
                    {"tool": "read_file", "success": true, "result": "x"},
                    {"tool": "grep", "success": false, "error": "boom"}
                ]
            }),
            true,
            12,
        );

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].tool_name, "read_file");
        assert_eq!(blocks[0].summary, "EXPLORE · 1 step");
        assert_eq!(blocks[0].detail_lines, vec!["Read a.rs"]);
        assert_eq!(blocks[1].tool_name, "grep");
        assert_eq!(blocks[1].summary, "EXPLORE · 1 step");
        assert_eq!(blocks[1].detail_lines, vec!["Search \"foo\" in ."]);
        assert_ne!(blocks[0].tool_name, "batch");
        assert_ne!(blocks[1].tool_name, "batch");
    }

    #[test]
    fn batch_normalizes_namespaced_child_tool_blocks() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
            "batch",
            json!({
                "tool_calls": [
                    {"tool": "functions.read_file", "parameters": {"path": "a.rs"}},
                    {"tool": "functions.grep", "parameters": {"pattern": "foo", "path": "."}}
                ]
            }),
            json!({
                "results": [
                    {"tool": "functions.read_file", "success": true, "result": "x"},
                    {"tool": "functions.grep", "success": true, "result": "match"}
                ]
            }),
            true,
            12,
        );

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].tool_name, "read_file");
        assert_eq!(blocks[0].summary, "EXPLORE · 1 step");
        assert_eq!(blocks[1].tool_name, "grep");
        assert_eq!(blocks[1].summary, "EXPLORE · 1 step");
    }

    #[test]
    fn namespaced_batch_expands_into_child_tool_blocks() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
            "functions.batch",
            json!({
                "tool_calls": [
                    {"tool": "functions.read_file", "parameters": {"path": "a.rs"}},
                    {"tool": "functions.grep", "parameters": {"pattern": "foo", "path": "."}}
                ]
            }),
            json!({
                "results": [
                    {"tool": "functions.read_file", "success": true, "result": "x"},
                    {"tool": "functions.grep", "success": true, "result": "match"}
                ]
            }),
            true,
            12,
        );

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].tool_name, "read_file");
        assert_eq!(blocks[1].tool_name, "grep");
    }

    #[test]
    fn projected_batch_details_expand_into_child_tool_blocks() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
            "batch",
            json!({
                "tool_calls": [
                    {"tool": "read_file", "parameters": {"path": "README.md"}},
                    {"tool": "search_web", "parameters": {"query": "OpenAI"}}
                ]
            }),
            json!({
                "summary": "All 2 tools executed successfully.",
                "details": [
                    {"tool": "read_file", "success": true, "duration_ms": 8},
                    {"tool": "search_web", "success": true, "duration_ms": 1300}
                ]
            }),
            true,
            1308,
        );

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].tool_name, "read_file");
        assert_eq!(blocks[0].summary, "EXPLORE · 1 step");
        assert_eq!(blocks[0].detail_lines, vec!["Read README.md"]);
        assert_eq!(blocks[1].tool_name, "search_web");
        assert_eq!(blocks[1].summary, "searched web for \"OpenAI\"");
    }

    #[test]
    fn read_file_labels_prefer_repo_relative_paths() {
        let mut state = ActivityState::default();
        let path = std::env::current_dir()
            .expect("cwd")
            .join("lash-cli/src/ui.rs");
        let blocks = state.blocks_for_tool_call(
            "read_file",
            json!({ "path": path }),
            json!("fn render() {}"),
            true,
            4,
        );

        assert_eq!(blocks[0].summary, "EXPLORE · 1 step");
        assert_eq!(blocks[0].detail_lines, vec!["Read lash-cli/src/ui.rs"]);
    }

    #[test]
    fn grep_labels_prefer_repo_relative_paths() {
        let mut state = ActivityState::default();
        let path = std::env::current_dir()
            .expect("cwd")
            .join("lash-cli/src/ui.rs");
        let blocks = state.blocks_for_tool_call(
            "grep",
            json!({ "pattern": "render_activity_block", "path": path }),
            json!("match"),
            true,
            4,
        );

        assert_eq!(
            blocks[0].detail_lines,
            vec!["Search \"render_activity_block\" in lash-cli/src/ui.rs"]
        );
    }

    #[test]
    fn compact_path_display_collapses_external_absolute_paths() {
        assert_eq!(
            compact_path_display("/tmp/std-template/spacetimedb/src/lib.rs"),
            "…/spacetimedb/src/lib.rs"
        );
    }

    #[test]
    fn write_stdin_uses_session_id_argument() {
        let mut state = ActivityState::default();
        state.shell_handles.insert("7".into(), "python3 -q".into());
        let blocks = state.blocks_for_tool_call(
            "write_stdin",
            json!({
                "session_id": 7,
                "chars": "print(2 + 2)\n"
            }),
            json!({
                "output": ">>> print(2 + 2)\n4\n>>> ",
                "session_id": 7,
                "wall_time_seconds": 0.01
            }),
            true,
            13,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].summary, "sent print(2 + 2) → python3 -q");
        assert!(blocks[0].detail_lines.iter().any(|line| line == "Handle 7"));
    }

    #[test]
    fn write_stdin_summary_uses_input_preview() {
        let mut state = ActivityState::default();
        state.blocks_for_tool_call(
            "exec_command",
            json!({"cmd":"python3 -q"}),
            json!({
                "session_id": 7,
                "output": ""
            }),
            true,
            1,
        );

        let wrote = state.blocks_for_tool_call(
            "write_stdin",
            json!({"session_id":7,"chars":"print(2 + 2)\n","yield_time_ms":1000}),
            json!({
                "exit_code": null,
                "session_id": 7,
                "output": ">>> print(2 + 2)\n4\n>>> ",
            }),
            true,
            7,
        );

        assert_eq!(wrote[0].summary, "sent print(2 + 2) → python3 -q");
    }

    #[test]
    fn tool_search_and_ask_show_specific_context() {
        let mut state = ActivityState::default();
        let search_blocks = state.blocks_for_tool_call(
            "search_tools",
            json!({"query":"planning"}),
            json!([
                {
                    "name":"update_plan",
                    "description":"Update the active execution plan with concrete steps."
                }
            ]),
            true,
            0,
        );
        let ask_blocks = state.blocks_for_tool_call(
            "ask",
            json!({"question":"Which environment should I use?","options":["staging","prod"]}),
            json!({"kind":"single","selection":"staging"}),
            true,
            0,
        );

        assert_eq!(search_blocks[0].summary, "searched tools for \"planning\"");
        assert_eq!(
            search_blocks[0].detail_lines,
            vec!["update_plan: Update the active execution plan with concrete steps."]
        );
        assert_eq!(ask_blocks[0].summary, "Question");
        assert_eq!(
            ask_blocks[0].detail_lines,
            vec!["Which environment should I use?", "1. staging", "2. prod",]
        );
    }

    #[test]
    fn ask_tool_result_omits_echoed_answer_and_note_lines() {
        let mut state = ActivityState::default();
        let ask_blocks = state.blocks_for_tool_call(
            "ask",
            json!({"question":"Which direction should I take?","options":["minimal","full"]}),
            json!({
                "kind":"single",
                "selection":"full",
                "note":"keep the transcript path stable"
            }),
            true,
            0,
        );

        assert_eq!(ask_blocks[0].summary, "Question");
        assert_eq!(
            ask_blocks[0].detail_lines,
            vec!["Which direction should I take?", "1. minimal", "2. full",]
        );
        assert!(matches!(
            ask_blocks[0].artifact.as_ref(),
            Some(ActivityArtifact::QuestionPanel(panel))
                if panel.options.len() == 2
                    && !panel.options[0].selected
                    && panel.options[1].selected
                    && panel.note.as_deref() == Some("keep the transcript path stable")
        ));
    }

    #[test]
    fn search_web_summary_keeps_query_in_primary_row() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
            "search_web",
            json!({ "query": "ratatui queue preview" }),
            json!({
                "answer": "Queue preview rendering is discussed in the ratatui docs.",
                "results": [
                    {
                        "title": "Ratatui queue preview guide",
                        "url": "https://ratatui.rs/guide/queue-preview",
                        "content": "..."
                    }
                ]
            }),
            true,
            18,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(
            blocks[0].summary,
            "searched web for \"ratatui queue preview\""
        );
        assert_eq!(
            blocks[0].detail_lines,
            vec![
                "Answer Queue preview rendering is discussed in the ratatui docs.".to_string(),
                "Ratatui queue preview guide · ratatui.rs/guide/queue-preview".to_string()
            ]
        );
    }

    #[test]
    fn agent_result_uses_child_status_and_token_usage() {
        let mut state = ActivityState::default();
        state.blocks_for_tool_call(
            "agent_call",
            json!({"prompt":"inspect queue rendering"}),
            json!({"id":"child-1","model":"gpt-5.4","model_variant":"high"}),
            true,
            3,
        );

        let blocks = state.blocks_for_tool_call(
            "agent_result",
            json!({"id":"child-1"}),
            json!({
                "result":"delegate result",
                "status":"interrupted",
                "session":{
                    "id":"child-1",
                    "parent_session_id":"root",
                    "task":"inspect queue rendering",
                    "model":"gpt-5.4",
                    "model_variant":"high",
                    "iterations":2,
                    "tool_calls":1,
                    "tool_call_details":[
                        {"tool":"read_file","success":true,"duration_ms":12}
                    ],
                    "token_usage":{
                        "input_tokens":101,
                        "output_tokens":22,
                        "cached_input_tokens":5,
                        "reasoning_tokens":7,
                        "total_tokens":135
                    }
                }
            }),
            true,
            12,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].status, ActivityStatus::Failed);
        assert_eq!(
            blocks[0].summary,
            "delegate stopped · inspect queue rendering"
        );
        assert_eq!(
            blocks[0].detail_lines,
            vec![
                "gpt-5.4 (high) · 2 iterations · 1 tool call".to_string(),
                "135 total tokens · 101 in · 22 out · 7 reasoning · 5 cached".to_string(),
            ]
        );
        assert!(blocks[0].children.is_empty());
    }

    #[test]
    fn agent_result_surfaces_child_error_details() {
        let mut state = ActivityState::default();
        state.blocks_for_tool_call(
            "agent_call",
            json!({"prompt":"inspect queue rendering"}),
            json!({"id":"child-1","model":"gpt-5.4-mini","model_variant":"low"}),
            true,
            3,
        );

        let blocks = state.blocks_for_tool_call(
            "agent_result",
            json!({"id":"child-1"}),
            json!({
                "result":"",
                "status":"failed",
                "error":"LLM error: Codex request failed with 400",
                "session":{
                    "id":"child-1",
                    "parent_session_id":"root",
                    "task":"inspect queue rendering",
                    "model":"gpt-5.4-mini",
                    "model_variant":"low",
                    "iterations":24,
                    "tool_calls":49,
                    "tool_call_details":[]
                }
            }),
            true,
            12,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].status, ActivityStatus::Failed);
        assert_eq!(
            blocks[0].summary,
            "delegate failed · inspect queue rendering"
        );
        assert_eq!(
            blocks[0].detail_lines,
            vec![
                "Error LLM error: Codex request failed with 400".to_string(),
                "gpt-5.4-mini (low) · 24 iterations · 49 tool calls".to_string(),
            ]
        );
    }
}
