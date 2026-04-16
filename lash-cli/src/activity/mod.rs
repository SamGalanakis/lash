use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

mod projector;
mod projectors;
mod shared;

pub use self::projector::{ProjectCtx, ToolProjector};
pub use self::projectors::edit::merge_edit_activity;
pub(crate) use self::projectors::edit::{patch_file_subject, patch_status_title};
pub use self::projectors::exploration::merge_exploration_activity;
use self::shared::activity_tool_name;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ActivityKind {
    Exploration,
    ShellCommand,
    ShellInteraction,
    WebSearch,
    WebFetch,
    Edit,
    Subagent,
    Parallel,
    Ask,
    GenericTool,
    Hidden,
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
    SnippetPreview(SnippetPreviewArtifact),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SnippetRenderMode {
    Markdown,
    Code,
    Text,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SnippetPreviewArtifact {
    pub title: Option<String>,
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub render_mode: SnippetRenderMode,
    pub language: Option<String>,
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

/// What the tool was invoked as. Describes the call, not what came back.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ActivityCall {
    pub kind: ActivityKind,
    pub tool_name: String,
    pub args: Value,
    /// Optional structured tag rendered as a distinct brand-weight span
    /// before `summary`. Currently unused — single-op explorations now
    /// promote the op straight onto the call line instead of using a tag
    /// wrapper. Kept as an affordance for future call kinds that want a
    /// short categorical label styled independently from the body.
    #[serde(default)]
    pub tag: Option<String>,
    /// Human-facing label for the call line (e.g. "Read README.md",
    /// "git status --short", "Explored").
    pub summary: String,
    pub extra: Option<ActivityExtra>,
}

/// What came back from the tool. Describes the result, not the invocation.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ActivityResult {
    pub status: ActivityStatus,
    /// Raw JSON result payload.
    pub raw: Value,
    /// Rendered preview lines shown under the call line.
    pub detail_lines: Vec<String>,
    pub artifact: Option<ActivityArtifact>,
}

/// A tool activity: a call paired with its result. The two sides used to be
/// flat fields on a single struct, which meant the renderer couldn't
/// address them independently (summary and result artifact competed for
/// the same block) and duration was burned into the summary string via a
/// `" · "` format. They're now nested so the renderer composes them as
/// separate styled regions.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ActivityBlock {
    pub call: ActivityCall,
    pub result: ActivityResult,
    pub duration_ms: u64,
    pub children: Vec<ActivityBlock>,
}

impl ActivityBlock {
    /// Construct an activity with the standard layout: call fields on one
    /// side, result fields on the other, no children. Use field-set helpers
    /// below for anything that needs customization beyond the defaults.
    pub fn new(
        kind: ActivityKind,
        tool_name: impl Into<String>,
        args: Value,
        summary: impl Into<String>,
        status: ActivityStatus,
        raw_result: Value,
        duration_ms: u64,
    ) -> Self {
        Self {
            call: ActivityCall {
                kind,
                tool_name: tool_name.into(),
                args,
                tag: None,
                summary: summary.into(),
                extra: None,
            },
            result: ActivityResult {
                status,
                raw: raw_result,
                detail_lines: Vec::new(),
                artifact: None,
            },
            duration_ms,
            children: Vec::new(),
        }
    }

    pub fn with_detail_lines(mut self, detail_lines: Vec<String>) -> Self {
        self.result.detail_lines = detail_lines;
        self
    }

    pub fn with_artifact(mut self, artifact: Option<ActivityArtifact>) -> Self {
        self.result.artifact = artifact;
        self
    }

    pub fn with_extra(mut self, extra: Option<ActivityExtra>) -> Self {
        self.call.extra = extra;
        self
    }
}

#[derive(Clone)]
pub struct ActivityState {
    shell_handles: HashMap<String, String>,
    /// Tool-name → projector registry, populated once at construction via
    /// `projectors::register_builtins`. Dispatch walks this map before
    /// falling through to the legacy match (during the refactor) or the
    /// generic projector (once the refactor is complete).
    registry: HashMap<&'static str, Arc<dyn ToolProjector>>,
}

impl Default for ActivityState {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ActivityState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `Arc<dyn ToolProjector>` doesn't implement `Debug` (and we
        // don't want to force that on every projector impl), so we
        // summarize the registry instead of recursing into it.
        f.debug_struct("ActivityState")
            .field("shell_handles", &self.shell_handles)
            .field("registry_len", &self.registry.len())
            .finish()
    }
}

impl ActivityState {
    pub fn new() -> Self {
        let mut state = Self {
            shell_handles: HashMap::new(),
            registry: HashMap::new(),
        };
        projectors::register_builtins(&mut state);
        state
    }

    /// Register a projector under every name it claims. Intended for
    /// built-in registration from `projectors::register_builtins`;
    /// exposing this publicly is a follow-up once the plugin extension
    /// surface lands.
    pub(super) fn register<P: ToolProjector + 'static>(&mut self, projector: P) {
        let arc: Arc<dyn ToolProjector> = Arc::new(projector);
        for name in arc.tool_names() {
            self.registry.insert(*name, Arc::clone(&arc));
        }
    }

    pub fn reset(&mut self) {
        self.shell_handles.clear();
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
        let status = if success {
            ActivityStatus::Completed
        } else {
            ActivityStatus::Failed
        };
        let mut ctx = ProjectCtx {
            name,
            args,
            result,
            success,
            duration_ms,
            shell_handles: &mut self.shell_handles,
        };
        if let Some(projector) = self.registry.get(name).cloned() {
            return projector.project(&mut ctx);
        }

        // Unregistered tool: fall back to the generic projector's
        // free function so we don't round-trip through the registry
        // (GenericProjector is already registered under `search_tools`).
        vec![projectors::generic::fallback_block(&mut ctx, status)]
    }

    fn blocks_for_batch_tool_call(&mut self, args: &Value, result: &Value) -> Vec<ActivityBlock> {
        let Some(entries) = result.get("results").and_then(|value| value.as_array()) else {
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
            .flat_map(|(index, item)| {
                let child_name = item
                    .get("tool")
                    .and_then(|value| value.as_str())
                    .or_else(|| {
                        calls
                            .get(index)
                            .and_then(|value| value.get("tool"))
                            .and_then(|value| value.as_str())
                    })
                    .unwrap_or("tool")
                    .to_string();
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

                self.blocks_for_tool_call(
                    &child_name,
                    child_args,
                    child_result,
                    child_success,
                    child_duration,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn batch_expands_into_child_tool_blocks() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
            "batch",
            json!({
                "tool_calls": [
                    {"tool": "read_file", "parameters": {"path": "a.rs"}},
                    {"tool": "grep", "parameters": {"query": "foo"}}
                ]
            }),
            json!({
                "results": [
                    {"tool": "read_file", "success": true, "result": "x"},
                    {"tool": "grep", "success": false, "error": "boom"}
                ]
            }),
            true,
            12,
        );

        // Single-op exploration activities promote the op straight onto
        // the call line. There are no detail lines and no `EXPLORE`
        // wrapper — that's now reserved for multi-op clusters.
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].call.tool_name, "read_file");
        assert_eq!(blocks[0].call.summary, "Read a.rs");
        assert!(blocks[0].result.detail_lines.is_empty());
        assert_eq!(blocks[1].call.tool_name, "grep");
        assert_eq!(blocks[1].call.summary, "Search \"foo\"");
        assert!(blocks[1].result.detail_lines.is_empty());
        assert_ne!(blocks[0].call.tool_name, "batch");
        assert_ne!(blocks[1].call.tool_name, "batch");
    }

    #[test]
    fn batch_normalizes_namespaced_child_tool_blocks() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
            "batch",
            json!({
                "tool_calls": [
                    {"tool": "functions.read_file", "parameters": {"path": "a.rs"}},
                    {"tool": "functions.grep", "parameters": {"query": "foo"}}
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
        assert_eq!(blocks[0].call.tool_name, "read_file");
        assert_eq!(blocks[0].call.summary, "Read a.rs");
        assert_eq!(blocks[1].call.tool_name, "grep");
        assert_eq!(blocks[1].call.summary, "Search \"foo\"");
    }

    #[test]
    fn namespaced_batch_expands_into_child_tool_blocks() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
            "functions.batch",
            json!({
                "tool_calls": [
                    {"tool": "functions.read_file", "parameters": {"path": "a.rs"}},
                    {"tool": "functions.grep", "parameters": {"query": "foo"}}
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
        assert_eq!(blocks[0].call.tool_name, "read_file");
        assert_eq!(blocks[1].call.tool_name, "grep");
    }

    #[test]
    fn projected_batch_results_expand_into_child_tool_blocks() {
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
                "results": [
                    {"tool": "read_file", "success": true, "duration_ms": 8, "result": "README body"},
                    {"tool": "search_web", "success": true, "duration_ms": 1300, "result": {"results": []}}
                ]
            }),
            true,
            1308,
        );

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].call.tool_name, "read_file");
        assert_eq!(blocks[0].call.summary, "Read README.md");
        assert!(blocks[0].result.detail_lines.is_empty());
        assert_eq!(blocks[1].call.tool_name, "search_web");
        assert_eq!(blocks[1].call.summary, "searched web for \"OpenAI\"");
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
        assert_eq!(blocks[0].call.summary, "sent print(2 + 2) → python3 -q");
        assert!(
            blocks[0]
                .result
                .detail_lines
                .iter()
                .any(|line| line == "Handle 7")
        );
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

        assert_eq!(wrote[0].call.summary, "sent print(2 + 2) → python3 -q");
    }

    #[test]
    fn write_stdin_empty_poll_without_output_is_suppressed() {
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

        let polled = state.blocks_for_tool_call(
            "write_stdin",
            json!({"session_id":7,"chars":"","yield_time_ms":1000}),
            json!({
                "session_id": 7,
                "output": "",
                "running": true
            }),
            true,
            2,
        );

        assert!(polled.is_empty());
    }

    #[test]
    fn write_stdin_empty_poll_with_output_is_kept() {
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

        let polled = state.blocks_for_tool_call(
            "write_stdin",
            json!({"session_id":7,"chars":"","yield_time_ms":1000}),
            json!({
                "session_id": 7,
                "output": ">>> ",
                "running": true
            }),
            true,
            2,
        );

        assert_eq!(polled.len(), 1);
        assert_eq!(polled[0].call.summary, "read python3 -q");
    }

    #[test]
    fn tool_search_and_ask_show_specific_context() {
        let mut state = ActivityState::default();
        let search_blocks = state.blocks_for_tool_call(
            "search_tools",
            json!({"query":"planning"}),
            json!([
                {
                    "name":"plan_exit",
                    "description":"Open the plan review prompt for the current plan file."
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

        assert_eq!(
            search_blocks[0].call.summary,
            "searched tools for \"planning\""
        );
        assert_eq!(
            search_blocks[0].result.detail_lines,
            vec!["plan_exit: Open the plan review prompt for the current plan file."]
        );
        assert_eq!(ask_blocks[0].call.summary, "Question");
        assert_eq!(
            ask_blocks[0].result.detail_lines,
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

        assert_eq!(ask_blocks[0].call.summary, "Question");
        assert_eq!(
            ask_blocks[0].result.detail_lines,
            vec!["Which direction should I take?", "1. minimal", "2. full",]
        );
        assert!(matches!(
            ask_blocks[0].result.artifact.as_ref(),
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
            json!({ "query": "terminal queue preview" }),
            json!({
                "answer": "Queue preview rendering is discussed in the terminal docs.",
                "results": [
                    {
                        "title": "Terminal queue preview guide",
                        "url": "https://example.com/guide/queue-preview",
                        "content": "..."
                    }
                ]
            }),
            true,
            18,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(
            blocks[0].call.summary,
            "searched web for \"terminal queue preview\""
        );
        assert_eq!(
            blocks[0].result.detail_lines,
            vec![
                "Answer Queue preview rendering is discussed in the terminal docs.".to_string(),
                "Terminal queue preview guide · example.com/guide/queue-preview".to_string()
            ]
        );
    }

    #[test]
    fn spawn_agent_projects_compact_headline_and_labeled_details() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
            "spawn_agent",
            json!({
                "task_name":"probe_repo_shape",
                "task":"In /home/sam/code/lash, inspect the repo shape only. Reply with the top-level summary."
            }),
            json!({
                "task_name":"probe_repo_shape",
                "path":"/root/probe_repo_shape",
                "capability":"low",
                "model":"gpt-5.4-mini",
                "model_variant":"low"
            }),
            true,
            12,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].call.summary, "spawn subagent · probe_repo_shape");
        assert_eq!(
            blocks[0].result.detail_lines,
            vec![
                "Task In /home/sam/code/lash, inspect the repo shape only. Reply with the top-level summary.".to_string(),
                "Path /root/probe_repo_shape".to_string(),
                "Profile low capability · gpt-5.4-mini".to_string(),
            ]
        );
    }

    #[test]
    fn wait_agent_task_started_uses_compact_name_and_wrapped_metadata() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
            "wait_agent",
            json!({"targets":["/root/probe_repo_shape"]}),
            json!({
                "timed_out": false,
                "events": [{
                    "kind":"task_started",
                    "path":"/root/probe_repo_shape",
                    "task":"In /home/sam/code/lash, inspect the repo shape only.",
                    "capability":"low",
                    "model":"gpt-5.4-mini",
                    "model_variant":"low"
                }]
            }),
            true,
            12,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].call.summary, "subagent started");
        assert_eq!(blocks[0].call.tag.as_deref(), Some("probe_repo_shape"));
        assert_eq!(
            blocks[0].result.detail_lines,
            vec![
                "Task In /home/sam/code/lash, inspect the repo shape only.".to_string(),
                "Path /root/probe_repo_shape".to_string(),
                "Profile low capability · gpt-5.4-mini".to_string(),
            ]
        );
    }

    #[test]
    fn wait_agent_projects_task_completion_and_token_usage() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
            "wait_agent",
            json!({"targets":["/root/inspect_queue"]}),
            json!({
                "timed_out": false,
                "events": [{
                    "kind":"task_completed",
                    "path":"/root/inspect_queue",
                    "task":"inspect queue rendering",
                    "status":"interrupted",
                    "result":"delegate result",
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
                }]
            }),
            true,
            12,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].result.status, ActivityStatus::Failed);
        assert_eq!(blocks[0].call.summary, "subagent stopped");
        assert_eq!(blocks[0].call.tag.as_deref(), Some("inspect_queue"));
        assert_eq!(
            blocks[0].result.detail_lines,
            vec![
                "Task inspect queue rendering".to_string(),
                "Path /root/inspect_queue".to_string(),
                "Run gpt-5.4 (high) · 2 iterations · 1 tool call".to_string(),
                "Tokens 135 total · 101 in · 22 out · 7 reasoning · 5 cached".to_string(),
            ]
        );
        assert!(blocks[0].children.is_empty());
    }

    #[test]
    fn wait_agent_surfaces_child_error_details() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
            "wait_agent",
            json!({"targets":["/root/inspect_queue"]}),
            json!({
                "timed_out": false,
                "events": [{
                    "kind":"task_completed",
                    "path":"/root/inspect_queue",
                    "task":"inspect queue rendering",
                    "status":"failed",
                    "result":"",
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
                }]
            }),
            true,
            12,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].result.status, ActivityStatus::Failed);
        assert_eq!(blocks[0].call.summary, "subagent failed");
        assert_eq!(blocks[0].call.tag.as_deref(), Some("inspect_queue"));
        assert_eq!(
            blocks[0].result.detail_lines,
            vec![
                "Task inspect queue rendering".to_string(),
                "Error LLM error: Codex request failed with 400".to_string(),
                "Path /root/inspect_queue".to_string(),
                "Run gpt-5.4-mini (low) · 24 iterations · 49 tool calls".to_string(),
            ]
        );
    }
}
