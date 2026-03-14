use std::collections::HashMap;

use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActivityKind {
    Exploration,
    ShellCommand,
    ShellInteraction,
    WebSearch,
    WebFetch,
    Edit,
    Delegate,
    PlanUpdate,
    SkillAction,
    Parallel,
    GenericTool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivityStatus {
    Completed,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExplorationOpKind {
    Read,
    Search,
    Glob,
    List,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExplorationOp {
    pub kind: ExplorationOpKind,
    pub subject: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActivityExtra {
    Exploration(Vec<ExplorationOp>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActivityArtifact {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatchFilePreview {
    pub path: String,
    pub from_path: Option<String>,
    pub status: String,
    pub added: usize,
    pub removed: usize,
    pub diff: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
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
    agent_handles: HashMap<String, String>,
}

impl ActivityState {
    pub fn reset(&mut self) {
        self.shell_handles.clear();
        self.agent_handles.clear();
    }

    pub fn blocks_for_tool_call(
        &mut self,
        name: &str,
        args: Value,
        result: Value,
        success: bool,
        duration_ms: u64,
    ) -> Vec<ActivityBlock> {
        vec![self.block_for_single_tool_call(name, args, result, success, duration_ms)]
    }

    fn block_for_single_tool_call(
        &mut self,
        name: &str,
        args: Value,
        result: Value,
        success: bool,
        duration_ms: u64,
    ) -> ActivityBlock {
        let status = if success {
            ActivityStatus::Completed
        } else {
            ActivityStatus::Failed
        };

        match name {
            "read_file" => {
                let subject = read_label(&result)
                    .unwrap_or_else(|| tool_arg_str(&args, "path").unwrap_or("file").to_string());
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
                let subject = tool_arg_str(&args, "path").unwrap_or(".").to_string();
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
            "shell" => {
                let command = tool_arg_str(&args, "command")
                    .unwrap_or("shell")
                    .to_string();
                if success && let Some(id) = tool_result_handle_id(&result) {
                    self.shell_handles.insert(id.to_string(), command.clone());
                }
                let detail_lines = if status == ActivityStatus::Failed {
                    tool_arg_str(&args, "workdir")
                        .map(|workdir| vec![format!("In {}", workdir)])
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                ActivityBlock {
                    kind: ActivityKind::ShellCommand,
                    status,
                    tool_name: name.to_string(),
                    summary: shell_start_summary(&command, status),
                    detail_lines,
                    duration_ms,
                    args,
                    result,
                    artifact: None,
                    children: Vec::new(),
                    extra: None,
                }
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
            "shell_wait" => {
                let handle_id = tool_arg_str(&args, "id").unwrap_or_default().to_string();
                let command = self
                    .shell_handles
                    .get(&handle_id)
                    .cloned()
                    .unwrap_or_else(|| format!("shell {}", handle_id));
                let running = shell_result_running(&result);
                let exit_code = shell_result_exit_code(&result);
                if !running {
                    self.shell_handles.remove(&handle_id);
                }
                let status = if !success || exit_code.is_some_and(|value| value != 0) {
                    ActivityStatus::Failed
                } else {
                    ActivityStatus::Completed
                };
                let mut detail_lines = Vec::new();
                if shell_result_timed_out(&result) {
                    detail_lines.push("Timed out; handle still running".to_string());
                }
                let artifact = shell_output_artifact(&result);
                ActivityBlock {
                    kind: ActivityKind::ShellCommand,
                    status,
                    tool_name: name.to_string(),
                    summary: shell_wait_summary(&command, &result, status),
                    detail_lines,
                    duration_ms,
                    args,
                    result,
                    artifact,
                    children: Vec::new(),
                    extra: None,
                }
            }
            "shell_read" => {
                let handle_id = tool_arg_str(&args, "id").unwrap_or_default().to_string();
                let command = self
                    .shell_handles
                    .get(&handle_id)
                    .cloned()
                    .unwrap_or_else(|| format!("shell {}", handle_id));
                let exit_code = shell_result_exit_code(&result);
                let status = if !success || exit_code.is_some_and(|value| value != 0) {
                    ActivityStatus::Failed
                } else {
                    ActivityStatus::Completed
                };
                let artifact = shell_output_artifact(&result);
                ActivityBlock {
                    kind: ActivityKind::ShellInteraction,
                    status,
                    tool_name: name.to_string(),
                    summary: shell_read_summary(&command, &result, status),
                    detail_lines: Vec::new(),
                    duration_ms,
                    args,
                    result,
                    artifact,
                    children: Vec::new(),
                    extra: None,
                }
            }
            "shell_write" => {
                let handle_id = tool_arg_str(&args, "id").unwrap_or_default().to_string();
                let command = self
                    .shell_handles
                    .get(&handle_id)
                    .cloned()
                    .unwrap_or_else(|| format!("shell {}", handle_id));
                let input = tool_arg_str(&args, "input")
                    .map(inline_text)
                    .unwrap_or_else(|| "stdin".to_string());
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
                let artifact = shell_output_artifact(&result);
                ActivityBlock {
                    kind: ActivityKind::ShellInteraction,
                    status,
                    tool_name: name.to_string(),
                    summary: shell_write_summary(&command, &input, &result, status),
                    detail_lines: if status == ActivityStatus::Failed {
                        vec![format!("Input {}", input)]
                    } else {
                        Vec::new()
                    },
                    duration_ms,
                    args,
                    result,
                    artifact,
                    children: Vec::new(),
                    extra: None,
                }
            }
            "shell_kill" => {
                let handle_id = tool_arg_str(&args, "id").unwrap_or_default().to_string();
                let command = self
                    .shell_handles
                    .remove(&handle_id)
                    .unwrap_or_else(|| format!("shell {}", handle_id));
                ActivityBlock {
                    kind: ActivityKind::ShellInteraction,
                    status,
                    tool_name: name.to_string(),
                    summary: format!("stopped {}", inline_text(&command)),
                    detail_lines: Vec::new(),
                    duration_ms,
                    args,
                    result,
                    artifact: None,
                    children: Vec::new(),
                    extra: None,
                }
            }
            "search_history" => ActivityBlock {
                kind: ActivityKind::GenericTool,
                status,
                tool_name: name.to_string(),
                summary: history_search_summary(&args),
                detail_lines: history_search_detail_lines(&result),
                duration_ms,
                args,
                artifact: None,
                children: Vec::new(),
                extra: None,
                result,
            },
            "search_mem" => ActivityBlock {
                kind: ActivityKind::GenericTool,
                status,
                tool_name: name.to_string(),
                summary: memory_search_summary(&args),
                detail_lines: memory_search_detail_lines(&result),
                duration_ms,
                args,
                artifact: None,
                children: Vec::new(),
                extra: None,
                result,
            },
            "mem_set" => ActivityBlock {
                kind: ActivityKind::GenericTool,
                status,
                tool_name: name.to_string(),
                summary: memory_set_summary(&args),
                detail_lines: memory_set_detail_lines(&args),
                duration_ms,
                args,
                artifact: None,
                children: Vec::new(),
                extra: None,
                result,
            },
            "mem_get" => ActivityBlock {
                kind: ActivityKind::GenericTool,
                status,
                tool_name: name.to_string(),
                summary: memory_get_summary(&args, &result),
                detail_lines: memory_entry_detail_lines(&result),
                duration_ms,
                args,
                artifact: None,
                children: Vec::new(),
                extra: None,
                result,
            },
            "mem_delete" => ActivityBlock {
                kind: ActivityKind::GenericTool,
                status,
                tool_name: name.to_string(),
                summary: format!(
                    "deleted memory {}",
                    tool_arg_str(&args, "key").unwrap_or("key")
                ),
                detail_lines: Vec::new(),
                duration_ms,
                args,
                artifact: None,
                children: Vec::new(),
                extra: None,
                result,
            },
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
                    self.agent_handles.insert(id.to_string(), task.clone());
                }
                ActivityBlock {
                    kind: ActivityKind::Delegate,
                    status,
                    tool_name: name.to_string(),
                    summary: format!("delegate · {}", inline_text(&task)),
                    detail_lines: Vec::new(),
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
                let meta = result.get("_sub_agent");
                let task = meta
                    .and_then(|value| value.get("task"))
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
                    .or_else(|| self.agent_handles.remove(&handle_id))
                    .unwrap_or_else(|| handle_id.clone());
                let mut detail_lines = Vec::new();
                if let Some(meta) = meta {
                    let tool_calls = meta.get("tool_calls").and_then(|value| value.as_u64());
                    let iterations = meta.get("iterations").and_then(|value| value.as_u64());
                    if tool_calls.is_some() || iterations.is_some() {
                        detail_lines.push(format!(
                            "Iterations: {} · tool calls: {}",
                            iterations.unwrap_or(0),
                            tool_calls.unwrap_or(0)
                        ));
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
                    status,
                    tool_name: name.to_string(),
                    summary: format!("delegate done · {}", inline_text(&task)),
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
                let task = self.agent_handles.remove(&handle_id).unwrap_or(handle_id);
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
            "ask" => ActivityBlock {
                kind: ActivityKind::GenericTool,
                status,
                tool_name: name.to_string(),
                summary: ask_summary(&args),
                detail_lines: ask_detail_lines(&result),
                duration_ms,
                args,
                artifact: None,
                children: Vec::new(),
                extra: None,
                result,
            },
            "update_plan" => ActivityBlock {
                kind: ActivityKind::PlanUpdate,
                status,
                tool_name: name.to_string(),
                summary: plan_update_summary(&args),
                detail_lines: args
                    .get("explanation")
                    .and_then(|value| value.as_str())
                    .map(|value| vec![value.to_string()])
                    .unwrap_or_default(),
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
            "search_skills" => ActivityBlock {
                kind: ActivityKind::SkillAction,
                status,
                tool_name: name.to_string(),
                summary: skill_search_summary(&args),
                detail_lines: skill_search_detail_lines(&result),
                duration_ms,
                args,
                artifact: None,
                children: Vec::new(),
                extra: None,
                result,
            },
            "load_skill" => ActivityBlock {
                kind: ActivityKind::SkillAction,
                status,
                tool_name: name.to_string(),
                summary: load_skill_summary(&args, &result),
                detail_lines: load_skill_detail_lines(&result),
                duration_ms,
                args,
                artifact: None,
                children: Vec::new(),
                extra: None,
                result,
            },
            "read_skill_file" => ActivityBlock {
                kind: ActivityKind::SkillAction,
                status,
                tool_name: name.to_string(),
                summary: read_skill_file_summary(&args),
                detail_lines: Vec::new(),
                duration_ms,
                args,
                artifact: text_preview_artifact(Some("Skill file"), &result),
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

fn plan_update_summary(args: &serde_json::Value) -> String {
    let Some(items) = args.get("plan").and_then(|value| value.as_array()) else {
        return "updated plan".to_string();
    };
    let completed = items
        .iter()
        .filter(|item| item.get("status").and_then(|value| value.as_str()) == Some("completed"))
        .count();
    let in_progress = items
        .iter()
        .filter(|item| item.get("status").and_then(|value| value.as_str()) == Some("in_progress"))
        .count();
    if in_progress > 0 {
        format!(
            "updated plan · {} steps, {} completed, {} in progress",
            items.len(),
            completed,
            in_progress
        )
    } else {
        format!(
            "updated plan · {} steps, {} completed",
            items.len(),
            completed
        )
    }
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
    target.children.push(incoming);
    rebuild_exploration_summary(target);
    true
}

fn rebuild_exploration_summary(block: &mut ActivityBlock) {
    let Some(ActivityExtra::Exploration(ops)) = block.extra.as_ref() else {
        return;
    };
    let detail_lines = exploration_detail_lines(ops);
    let summary_bits = detail_lines
        .iter()
        .map(|line| line.to_ascii_lowercase())
        .collect::<Vec<_>>();
    block.summary = format!("explored · {}", summary_bits.join(", "));
    block.detail_lines = detail_lines;
}

fn semantic_tool_summary(name: &str, args: &Value) -> String {
    match name {
        "read_file" => tool_arg_str(args, "path")
            .map(|path| {
                let mut label = format!("read {}", path);
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
        "ls" => format!("list {}", tool_arg_str(args, "path").unwrap_or(".")),
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
        "shell" => tool_arg_str(args, "command")
            .map(|cmd| format!("shell {}", inline_text(cmd)))
            .unwrap_or_else(|| "shell".to_string()),
        "shell_wait" => "wait for shell".to_string(),
        "shell_read" => "read shell output".to_string(),
        "shell_write" => "write to shell".to_string(),
        "shell_kill" => "kill shell".to_string(),
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

fn exploration_detail_lines(ops: &[ExplorationOp]) -> Vec<String> {
    let mut reads = Vec::new();
    let mut searches = Vec::new();
    let mut globs = Vec::new();
    let mut lists = Vec::new();

    for op in ops {
        match op.kind {
            ExplorationOpKind::Read => reads.push(op.subject.clone()),
            ExplorationOpKind::Search => searches.push(op.subject.clone()),
            ExplorationOpKind::Glob => globs.push(op.subject.clone()),
            ExplorationOpKind::List => lists.push(op.subject.clone()),
        }
    }

    let mut lines = Vec::new();
    if !searches.is_empty() {
        lines.push(format!("Search {}", searches.join(", ")));
    }
    if !reads.is_empty() {
        lines.push(format!("Read {}", dedupe_preserve_order(reads).join(", ")));
    }
    if !globs.is_empty() {
        lines.push(format!("Glob {}", dedupe_preserve_order(globs).join(", ")));
    }
    if !lists.is_empty() {
        lines.push(format!("List {}", dedupe_preserve_order(lists).join(", ")));
    }
    lines
}

fn dedupe_preserve_order(values: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for value in values {
        if seen.insert(value.clone()) {
            out.push(value);
        }
    }
    out
}

fn grep_label(args: &Value) -> String {
    let pattern = tool_arg_str(args, "pattern").unwrap_or("pattern");
    if let Some(path) = tool_arg_str(args, "path") {
        format!("{:?} in {}", pattern, path)
    } else {
        format!("{:?}", pattern)
    }
}

fn glob_label(args: &Value) -> String {
    let pattern = tool_arg_str(args, "pattern").unwrap_or("*");
    if let Some(path) = tool_arg_str(args, "path") {
        format!("{} in {}", pattern, path)
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
        return Some(path.to_string());
    }
    if let Some(path) = first_line.split_whitespace().next()
        && path.contains('/')
    {
        return Some(path.to_string());
    }
    None
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
    result.get("id").and_then(|value| value.as_str())
}

fn tool_result_shell_id(result: &Value) -> Option<String> {
    result
        .get("id")
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

fn tool_arg_shell_id(args: &Value) -> Option<String> {
    args.get("id")
        .and_then(|value| value.as_str())
        .map(str::to_string)
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
    tool_arg_str(args, "question")
        .map(|question| format!("asked user · {}", inline_text(question)))
        .unwrap_or_else(|| "asked user".to_string())
}

fn ask_detail_lines(result: &Value) -> Vec<String> {
    result
        .as_str()
        .filter(|answer| !answer.trim().is_empty())
        .map(|answer| vec![format!("Answer {}", inline_snippet(answer, 72))])
        .unwrap_or_default()
}

fn tool_search_summary(args: &Value) -> String {
    tool_arg_str(args, "query")
        .map(|query| format!("searched tools for {:?}", inline_text(query)))
        .unwrap_or_else(|| "browsed tools".to_string())
}

fn tool_search_detail_lines(result: &Value) -> Vec<String> {
    named_description_detail_lines(result, 4)
}

fn skill_search_summary(args: &Value) -> String {
    tool_arg_str(args, "query")
        .map(|query| format!("searched skills for {:?}", inline_text(query)))
        .unwrap_or_else(|| "browsed skills".to_string())
}

fn skill_search_detail_lines(result: &Value) -> Vec<String> {
    named_description_detail_lines(result, 4)
}

fn load_skill_summary(args: &Value, result: &Value) -> String {
    result
        .get("name")
        .and_then(|value| value.as_str())
        .or_else(|| tool_arg_str(args, "name"))
        .map(|name| format!("loaded skill {name}"))
        .unwrap_or_else(|| "loaded skill".to_string())
}

fn load_skill_detail_lines(result: &Value) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(description) = result.get("description").and_then(|value| value.as_str())
        && !description.trim().is_empty()
    {
        lines.push(inline_snippet(description, 72));
    }
    if let Some(file_count) = result.get("file_count").and_then(|value| value.as_u64()) {
        lines.push(format!("{file_count} supporting file(s)"));
    }
    lines
}

fn read_skill_file_summary(args: &Value) -> String {
    match (tool_arg_str(args, "name"), tool_arg_str(args, "path")) {
        (Some(name), Some(path)) => format!("read skill file {name}/{path}"),
        (_, Some(path)) => format!("read skill file {path}"),
        _ => "read skill file".to_string(),
    }
}

fn shell_start_summary(command: &str, status: ActivityStatus) -> String {
    if status == ActivityStatus::Failed {
        format!("failed {}", inline_text(command))
    } else {
        format!("started {}", inline_text(command))
    }
}

fn shell_wait_summary(command: &str, result: &Value, status: ActivityStatus) -> String {
    if status == ActivityStatus::Failed {
        return format!("failed {}", inline_text(command));
    }
    if shell_result_running(result) {
        return format!("still running {}", inline_text(command));
    }
    format!("finished {}", inline_text(command))
}

fn shell_read_summary(command: &str, result: &Value, status: ActivityStatus) -> String {
    if status == ActivityStatus::Failed {
        return format!("failed {}", inline_text(command));
    }
    if shell_result_output(result).is_some_and(|text| !text.trim().is_empty()) {
        return format!("read {}", inline_text(command));
    }
    if shell_result_running(result) {
        return format!("polled {}", inline_text(command));
    }
    format!("finished {}", inline_text(command))
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

fn history_search_summary(args: &Value) -> String {
    tool_arg_str(args, "query")
        .map(|query| format!("searched history for {:?}", inline_text(query)))
        .unwrap_or_else(|| "searched history".to_string())
}

fn history_search_detail_lines(result: &Value) -> Vec<String> {
    result
        .as_array()
        .map(|items| {
            items
                .iter()
                .take(3)
                .filter_map(|item| {
                    let turn = item.get("turn").and_then(|value| value.as_i64())?;
                    let preview = item.get("preview").and_then(|value| value.as_str())?;
                    Some(format!("Turn {}: {}", turn, inline_snippet(preview, 72)))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn memory_search_summary(args: &Value) -> String {
    match tool_arg_str(args, "query").map(str::trim) {
        Some("") | None => "listed memory".to_string(),
        Some(query) => format!("searched memory for {:?}", inline_text(query)),
    }
}

fn memory_search_detail_lines(result: &Value) -> Vec<String> {
    result
        .as_array()
        .map(|items| {
            items
                .iter()
                .take(3)
                .filter_map(|item| {
                    let key = item.get("key").and_then(|value| value.as_str())?;
                    let description = item
                        .get("description")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default();
                    let value = item
                        .get("value")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default();
                    Some(memory_entry_line(key, description, value))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn memory_set_summary(args: &Value) -> String {
    format!(
        "saved memory {}",
        tool_arg_str(args, "key").unwrap_or("key")
    )
}

fn memory_set_detail_lines(args: &Value) -> Vec<String> {
    let description = tool_arg_str(args, "description").unwrap_or_default();
    let value = args
        .get("value")
        .and_then(|value| value.as_str())
        .unwrap_or(description);
    let key = tool_arg_str(args, "key").unwrap_or_default();
    if key.is_empty() {
        Vec::new()
    } else {
        vec![memory_entry_line(key, description, value)]
    }
}

fn memory_get_summary(args: &Value, result: &Value) -> String {
    let key = tool_arg_str(args, "key").unwrap_or("key");
    if result.is_null() {
        format!("memory {} missing", key)
    } else {
        format!("loaded memory {}", key)
    }
}

fn memory_entry_detail_lines(result: &Value) -> Vec<String> {
    let key = result.get("key").and_then(|value| value.as_str());
    let description = result
        .get("description")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let value = result
        .get("value")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    key.map(|key| vec![memory_entry_line(key, description, value)])
        .unwrap_or_default()
}

fn shell_result_running(result: &Value) -> bool {
    result
        .get("running")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

fn shell_result_exit_code(result: &Value) -> Option<i64> {
    result.get("exit_code").and_then(|value| value.as_i64())
}

fn shell_result_timed_out(result: &Value) -> bool {
    result
        .get("timed_out")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
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

    Some(if files.len() == 1 {
        let file = &files[0];
        let path = match &file.from_path {
            Some(from_path) => format!("{from_path} → {}", file.path),
            None => file.path.clone(),
        };
        format!(
            "{} {} {}",
            patch_status_verb(&file.status),
            path,
            patch_count_suffix(file.added, file.removed)
        )
    } else {
        format!(
            "{} {} files {}",
            patch_group_verb(files.as_slice()),
            files.len(),
            patch_count_suffix(total_added, total_removed)
        )
    })
}

fn patch_group_verb(files: &[PatchFilePreview]) -> &'static str {
    let Some(first) = files.first() else {
        return "edited";
    };
    if files.iter().all(|file| file.status == first.status) {
        patch_status_verb(&first.status)
    } else {
        "edited"
    }
}

fn patch_status_verb(status: &str) -> &'static str {
    match status {
        "added" => "added",
        "deleted" => "deleted",
        "moved" => "moved",
        _ => "edited",
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

fn memory_entry_line(key: &str, description: &str, value: &str) -> String {
    let content = if !value.trim().is_empty() {
        value
    } else {
        description
    };
    if content.trim().is_empty() {
        key.to_string()
    } else {
        format!("{key}: {}", inline_snippet(content, 72))
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

        assert_eq!(blocks[0].summary, "edited lash-cli/src/ui.rs (+3 -1)");
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

        assert_eq!(blocks[0].summary, "moved old.rs → new.rs (+2 -2)");
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
    fn shell_lifecycle_summaries_are_distinct() {
        let mut state = ActivityState::default();
        let start = state.blocks_for_tool_call(
            "shell",
            json!({"command":"mktemp -d /tmp/lash_tool_test_XXXXXX"}),
            json!({
                "__handle__": "shell",
                "id": "sh-1",
                "pid": 42,
                "command": "mktemp -d /tmp/lash_tool_test_XXXXXX",
                "workdir": "/tmp",
                "timeout_ms": null,
                "login": false,
                "running": true
            }),
            true,
            1,
        );
        let waited = state.blocks_for_tool_call(
            "shell_wait",
            json!({"id":"sh-1","timeout_ms":3000}),
            json!({
                "id":"sh-1",
                "running": false,
                "exit_code": 0,
                "output": "/tmp/lash_tool_test_abc123\n",
                "timed_out": false
            }),
            true,
            3,
        );

        assert_eq!(
            start[0].summary,
            "started mktemp -d /tmp/lash_tool_test_XXXXXX"
        );
        assert_eq!(
            waited[0].summary,
            "finished mktemp -d /tmp/lash_tool_test_XXXXXX"
        );
    }

    #[test]
    fn shell_write_summary_uses_input_preview() {
        let mut state = ActivityState::default();
        state.blocks_for_tool_call(
            "shell",
            json!({"command":"python3 -q"}),
            json!({
                "__handle__": "shell",
                "id": "py-1",
                "pid": 77,
                "command": "python3 -q",
                "workdir": "/tmp",
                "timeout_ms": null,
                "login": false,
                "running": true
            }),
            true,
            1,
        );

        let wrote = state.blocks_for_tool_call(
            "shell_write",
            json!({"id":"py-1","input":"print(2 + 2)\n","timeout_ms":1000}),
            json!({
                "id":"py-1",
                "running": true,
                "exit_code": null,
                "output": ">>> print(2 + 2)\n4\n>>> ",
                "timed_out": false
            }),
            true,
            7,
        );

        assert_eq!(wrote[0].summary, "sent print(2 + 2) → python3 -q");
    }

    #[test]
    fn memory_tools_show_keys_and_snippets() {
        let mut state = ActivityState::default();

        let set_blocks = state.blocks_for_tool_call(
            "mem_set",
            json!({
                "key": "repo_convention",
                "description": "Use snake_case for tool names",
                "value": "Use snake_case for tool names"
            }),
            json!(null),
            true,
            0,
        );
        let search_blocks = state.blocks_for_tool_call(
            "search_mem",
            json!({"query":"snake_case"}),
            json!([
                {
                    "key":"repo_convention",
                    "description":"Use snake_case for tool names",
                    "value":"Use snake_case for tool names",
                    "turn": 2,
                    "score": 1.0,
                    "field_hits": ["value"]
                }
            ]),
            true,
            0,
        );

        assert_eq!(set_blocks[0].summary, "saved memory repo_convention");
        assert_eq!(
            set_blocks[0].detail_lines,
            vec!["repo_convention: Use snake_case for tool names"]
        );
        assert_eq!(
            search_blocks[0].summary,
            "searched memory for \"snake_case\""
        );
        assert_eq!(
            search_blocks[0].detail_lines,
            vec!["repo_convention: Use snake_case for tool names"]
        );
    }

    #[test]
    fn skill_search_and_load_show_match_details() {
        let mut state = ActivityState::default();
        let search_blocks = state.blocks_for_tool_call(
            "search_skills",
            json!({"query":"frontend"}),
            json!([
                {
                    "name":"frontend-design",
                    "description":"Create distinctive production-grade frontend interfaces.",
                    "file_count": 3,
                    "score": 1.0
                }
            ]),
            true,
            0,
        );
        let load_blocks = state.blocks_for_tool_call(
            "load_skill",
            json!({"name":"frontend-design"}),
            json!({
                "name":"frontend-design",
                "description":"Create distinctive production-grade frontend interfaces.",
                "instructions":"...",
                "files":["refs.md", "template.html"],
                "file_count": 2
            }),
            true,
            0,
        );

        assert_eq!(search_blocks[0].summary, "searched skills for \"frontend\"");
        assert_eq!(
            search_blocks[0].detail_lines,
            vec!["frontend-design: Create distinctive production-grade frontend interfaces."]
        );
        assert_eq!(load_blocks[0].summary, "loaded skill frontend-design");
        assert_eq!(
            load_blocks[0].detail_lines,
            vec![
                "Create distinctive production-grade frontend interfaces.",
                "2 supporting file(s)"
            ]
        );
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
            json!("staging"),
            true,
            0,
        );

        assert_eq!(search_blocks[0].summary, "searched tools for \"planning\"");
        assert_eq!(
            search_blocks[0].detail_lines,
            vec!["update_plan: Update the active execution plan with concrete steps."]
        );
        assert_eq!(
            ask_blocks[0].summary,
            "asked user · Which environment should I use?"
        );
        assert_eq!(ask_blocks[0].detail_lines, vec!["Answer staging"]);
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
}
