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
    TaskAction,
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
    DiffPreview { title: String, diff: String },
    TextPreview { title: Option<String>, text: String },
    SourceList { title: String, items: Vec<String> },
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
        if name == "shell_command" {
            return Vec::new();
        }
        if name == "batch" {
            return self.blocks_for_batch(args, result, success, duration_ms);
        }

        vec![self.block_for_single_tool_call(name, args, result, success, duration_ms)]
    }

    fn blocks_for_batch(
        &mut self,
        args: Value,
        result: Value,
        success: bool,
        duration_ms: u64,
    ) -> Vec<ActivityBlock> {
        let Some(results) = result.get("results").and_then(|value| value.as_array()) else {
            return vec![self.block_for_single_tool_call(
                "batch",
                args,
                result,
                success,
                duration_ms,
            )];
        };

        let mut children = Vec::new();
        for child in results {
            let Some(tool) = child.get("tool").and_then(|value| value.as_str()) else {
                continue;
            };
            let child_args = child
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            let child_result = child.get("result").cloned().unwrap_or(Value::Null);
            let child_success = child
                .get("success")
                .and_then(|value| value.as_bool())
                .unwrap_or(true);
            let child_duration_ms = child
                .get("duration_ms")
                .and_then(|value| value.as_u64())
                .unwrap_or(0);
            children.extend(self.blocks_for_tool_call(
                tool,
                child_args,
                child_result,
                child_success,
                child_duration_ms,
            ));
        }

        let visible_children: Vec<ActivityBlock> = children
            .into_iter()
            .filter(|child| {
                !tool_call_hidden(&child.tool_name, child.status == ActivityStatus::Completed)
            })
            .collect();

        visible_children
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
                let mut detail_lines = vec![format!("Command: {}", inline_text(&command))];
                if let Some(workdir) = tool_arg_str(&args, "workdir") {
                    detail_lines.push(format!("Workdir: {}", workdir));
                }
                ActivityBlock {
                    kind: ActivityKind::ShellCommand,
                    status,
                    tool_name: name.to_string(),
                    summary: format!("started {}", inline_text(&command)),
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
                let running_session = result
                    .get("session_id")
                    .and_then(|value| value.as_i64())
                    .map(|value| value.to_string());
                if let Some(session_id) = running_session.as_ref() {
                    self.shell_handles
                        .insert(session_id.clone(), command.clone());
                }
                let exit_code = result.get("exit_code").and_then(|value| value.as_i64());
                let status = if !success || exit_code.is_some_and(|value| value != 0) {
                    ActivityStatus::Failed
                } else {
                    ActivityStatus::Completed
                };
                let mut detail_lines = vec![format!("Command: {}", inline_text(&command))];
                if let Some(workdir) = tool_arg_str(&args, "workdir") {
                    detail_lines.push(format!("Workdir: {}", workdir));
                }
                if let Some(exit_code) = exit_code {
                    detail_lines.push(format!("Exit code: {}", exit_code));
                }
                if let Some(session_id) = running_session {
                    detail_lines.push(format!("Session: {}", session_id));
                }
                let artifact = result
                    .get("output")
                    .and_then(|value| value.as_str())
                    .filter(|text| !text.trim().is_empty())
                    .map(|text| ActivityArtifact::TextPreview {
                        title: Some("Shell output".to_string()),
                        text: text.to_string(),
                    });
                let summary = if status == ActivityStatus::Failed {
                    format!("failed {}", inline_text(&command))
                } else if result.get("session_id").is_some() {
                    format!("started {}", inline_text(&command))
                } else {
                    format!("ran {}", inline_text(&command))
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
                let session_id = args
                    .get("session_id")
                    .and_then(|value| value.as_i64())
                    .map(|value| value.to_string())
                    .unwrap_or_default();
                let command = self
                    .shell_handles
                    .get(&session_id)
                    .cloned()
                    .unwrap_or_else(|| format!("exec session {}", session_id));
                let exit_code = result.get("exit_code").and_then(|value| value.as_i64());
                if exit_code.is_some() {
                    self.shell_handles.remove(&session_id);
                } else if !session_id.is_empty() {
                    self.shell_handles
                        .entry(session_id.clone())
                        .or_insert_with(|| command.clone());
                }
                let status = if !success || exit_code.is_some_and(|value| value != 0) {
                    ActivityStatus::Failed
                } else {
                    ActivityStatus::Completed
                };
                let mut detail_lines = vec![format!("Command: {}", inline_text(&command))];
                let chars = tool_arg_str(&args, "chars").unwrap_or("");
                if !chars.is_empty() {
                    detail_lines.push(format!("Input: {}", inline_text(chars)));
                }
                if let Some(exit_code) = exit_code {
                    detail_lines.push(format!("Exit code: {}", exit_code));
                } else if !session_id.is_empty() {
                    detail_lines.push(format!("Session: {}", session_id));
                }
                let artifact = result
                    .get("output")
                    .and_then(|value| value.as_str())
                    .filter(|text| !text.trim().is_empty())
                    .map(|text| ActivityArtifact::TextPreview {
                        title: Some("Shell output".to_string()),
                        text: text.to_string(),
                    });
                let summary = if status == ActivityStatus::Failed {
                    format!("failed {}", inline_text(&command))
                } else if exit_code.is_some() {
                    format!("ran {}", inline_text(&command))
                } else if chars.is_empty() {
                    format!("polled {}", inline_text(&command))
                } else {
                    format!("sent input · {}", inline_text(&command))
                };
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
                if success {
                    self.shell_handles.remove(&handle_id);
                }
                let output = result.as_str().unwrap_or("").to_string();
                let artifact = (!output.trim().is_empty()).then(|| ActivityArtifact::TextPreview {
                    title: Some("Shell output".to_string()),
                    text: output,
                });
                ActivityBlock {
                    kind: ActivityKind::ShellCommand,
                    status,
                    tool_name: name.to_string(),
                    summary: format!("ran {}", inline_text(&command)),
                    detail_lines: vec![format!("Command: {}", inline_text(&command))],
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
                let output = result.as_str().unwrap_or("").to_string();
                let artifact = (!output.trim().is_empty()).then(|| ActivityArtifact::TextPreview {
                    title: Some("Shell output".to_string()),
                    text: output,
                });
                ActivityBlock {
                    kind: ActivityKind::ShellInteraction,
                    status,
                    tool_name: name.to_string(),
                    summary: format!("read shell output · {}", inline_text(&command)),
                    detail_lines: vec![format!("Command: {}", inline_text(&command))],
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
                ActivityBlock {
                    kind: ActivityKind::ShellInteraction,
                    status,
                    tool_name: name.to_string(),
                    summary: format!("sent shell input · {}", inline_text(&command)),
                    detail_lines: vec![format!("Input: {}", input)],
                    duration_ms,
                    args,
                    result,
                    artifact: None,
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
                    summary: format!("killed {}", inline_text(&command)),
                    detail_lines: vec![format!("Command: {}", inline_text(&command))],
                    duration_ms,
                    args,
                    result,
                    artifact: None,
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
                    summary: format!("searched web {:?}", query),
                    detail_lines: vec![format!("Query: {}", query)],
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
                    summary: format!("fetched {}", url),
                    detail_lines: vec![format!("URL: {}", url)],
                    duration_ms,
                    args,
                    result,
                    artifact,
                    children: Vec::new(),
                    extra: None,
                }
            }
            "apply_patch" => {
                let summary = result
                    .get("summary")
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| semantic_tool_summary(name, &args));
                let artifact = result
                    .get("diff")
                    .and_then(|value| value.as_str())
                    .filter(|diff| !diff.trim().is_empty())
                    .map(|diff| ActivityArtifact::DiffPreview {
                        title: "Diff".to_string(),
                        diff: diff.to_string(),
                    });
                ActivityBlock {
                    kind: ActivityKind::Edit,
                    status,
                    tool_name: name.to_string(),
                    summary,
                    detail_lines: patch_detail_lines(&result),
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
                    summary: format!("delegated {}", inline_text(&task)),
                    detail_lines: vec![format!("Task: {}", task)],
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
                let mut detail_lines = vec![format!("Task: {}", task)];
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
                    summary: format!("delegate result · {}", inline_text(&task)),
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
                    summary: format!("killed delegate {}", inline_text(&task)),
                    detail_lines: vec![format!("Task: {}", task)],
                    duration_ms,
                    args,
                    result,
                    artifact: None,
                    children: Vec::new(),
                    extra: None,
                }
            }
            "create_task" | "update_task" | "claim_task" | "delete_task" | "get_task" | "tasks"
            | "tasks_summary" => ActivityBlock {
                kind: ActivityKind::TaskAction,
                status,
                tool_name: name.to_string(),
                summary: semantic_tool_summary(name, &args),
                detail_lines: Vec::new(),
                duration_ms,
                args,
                result,
                artifact: None,
                children: Vec::new(),
                extra: None,
            },
            "update_plan" => ActivityBlock {
                kind: ActivityKind::TaskAction,
                status,
                tool_name: name.to_string(),
                summary: result
                    .get("summary")
                    .and_then(|value| value.as_str())
                    .unwrap_or("updated plan")
                    .to_string(),
                detail_lines: result
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
            "skills" | "search_skills" | "load_skill" | "read_skill_file" | "list_tools"
            | "search_tools" => ActivityBlock {
                kind: ActivityKind::SkillAction,
                status,
                tool_name: name.to_string(),
                summary: semantic_tool_summary(name, &args),
                detail_lines: Vec::new(),
                duration_ms,
                args,
                artifact: text_preview_artifact(None, &result),
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

pub fn tool_call_hidden(_name: &str, _success: bool) -> bool {
    false
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

pub fn rebuild_exploration_summary(block: &mut ActivityBlock) {
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

pub fn semantic_tool_summary(name: &str, args: &Value) -> String {
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
            .map(|cmd| format!("run {}", inline_text(cmd)))
            .unwrap_or_else(|| "run command".to_string()),
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
            .map(|url| format!("fetch {}", url))
            .unwrap_or_else(|| "fetch url".to_string()),
        "search_web" => tool_arg_str(args, "query")
            .map(|query| format!("search {:?}", query))
            .unwrap_or_else(|| "search web".to_string()),
        "agent_call" => tool_arg_str(args, "task")
            .or_else(|| tool_arg_str(args, "prompt"))
            .map(|task| format!("delegate {}", inline_text(task)))
            .unwrap_or_else(|| "delegate task".to_string()),
        "agent_result" => "collect delegate result".to_string(),
        "agent_output" => "stream delegate output".to_string(),
        "agent_kill" => "kill delegate".to_string(),
        "create_task" => tool_arg_str(args, "description")
            .or_else(|| tool_arg_str(args, "subject"))
            .map(|desc| format!("create task {}", inline_text(desc)))
            .unwrap_or_else(|| "create task".to_string()),
        "update_task" | "claim_task" | "delete_task" | "get_task" => tool_arg_str(args, "id")
            .map(|id| format!("{} {}", human_tool_name(name), id))
            .unwrap_or_else(|| human_tool_name(name).to_string()),
        "tasks" | "tasks_summary" | "list_tools" | "search_tools" | "search_skills" | "skills" => {
            human_tool_name(name).to_string()
        }
        "load_skill" | "read_skill_file" => tool_arg_str(args, "path")
            .map(|path| format!("{} {}", human_tool_name(name), path))
            .unwrap_or_else(|| human_tool_name(name).to_string()),
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

fn tool_arg_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key)
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
}

fn patch_detail_lines(result: &Value) -> Vec<String> {
    result
        .get("files")
        .and_then(|value| value.as_array())
        .map(|files| {
            files
                .iter()
                .take(5)
                .filter_map(|file| {
                    let path = file.get("path").and_then(|value| value.as_str())?;
                    let status = file
                        .get("status")
                        .and_then(|value| value.as_str())
                        .unwrap_or("modified");
                    let mut line = format!("{}: {}", capitalize(status), path);
                    if let Some(from_path) = file.get("from_path").and_then(|value| value.as_str())
                    {
                        line.push_str(&format!(" (from {})", from_path));
                    }
                    Some(line)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn inline_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn human_tool_name(name: &str) -> &'static str {
    match name {
        "read_file" => "read",
        "apply_patch" => "patch",
        "grep" => "grep",
        "glob" => "glob",
        "ls" => "list",
        "exec_command" => "run",
        "write_stdin" => "write stdin",
        "shell" => "shell",
        "shell_wait" => "wait",
        "shell_read" => "read shell",
        "shell_write" => "write shell",
        "shell_kill" => "kill shell",
        "fetch_url" => "fetch",
        "search_web" => "search web",
        "agent_call" => "delegate",
        "agent_result" => "delegate result",
        "agent_output" => "delegate output",
        "agent_kill" => "kill delegate",
        "create_task" => "create task",
        "update_task" => "update task",
        "claim_task" => "claim task",
        "delete_task" => "delete task",
        "get_task" => "task",
        "tasks" => "tasks",
        "tasks_summary" => "tasks summary",
        "list_tools" => "list tools",
        "search_tools" => "search tools",
        "search_skills" => "search skills",
        "skills" => "skills",
        "load_skill" => "load skill",
        "read_skill_file" => "read skill file",
        _ => "tool",
    }
}

fn capitalize(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn batch_with_exploration_children_are_flattened() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
            "batch",
            json!({}),
            json!({
                "__type__": "batch_results",
                "results": [
                    {
                        "tool": "grep",
                        "parameters": {"pattern": "ctx", "path": "lash-cli/src"},
                        "success": true,
                        "result": "match",
                        "duration_ms": 12
                    },
                    {
                        "tool": "read_file",
                        "parameters": {"path": "lash-cli/src/ui.rs"},
                        "success": true,
                        "result": "==> lash-cli/src/ui.rs <==\nline",
                        "duration_ms": 8
                    }
                ]
            }),
            true,
            20,
        );

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].kind, ActivityKind::Exploration);
        assert_eq!(blocks[1].kind, ActivityKind::Exploration);
    }

    #[test]
    fn batch_with_mixed_children_is_flattened() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
            "batch",
            json!({}),
            json!({
                "__type__": "batch_results",
                "results": [
                    {
                        "tool": "read_file",
                        "parameters": {"path": "lash-cli/src/ui.rs"},
                        "success": true,
                        "result": "==> lash-cli/src/ui.rs <==\nline",
                        "duration_ms": 8
                    },
                    {
                        "tool": "apply_patch",
                        "parameters": {"input": "*** Begin Patch\n*** Update File: lash-cli/src/ui.rs\n@@\n-old\n+new\n*** End Patch"},
                        "success": true,
                        "result": {
                            "summary": "Applied patch to 1 file",
                            "files": [{"path": "lash-cli/src/ui.rs", "status": "modified"}],
                            "diff": "--- a\n+++ b"
                        },
                        "duration_ms": 15
                    }
                ]
            }),
            true,
            30,
        );

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].kind, ActivityKind::Exploration);
        assert_eq!(blocks[1].kind, ActivityKind::Edit);
    }
}
