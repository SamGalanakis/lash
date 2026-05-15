use std::collections::HashMap;
use std::sync::Arc;

use lash_subagents::SubagentHost;
use serde_json::Value;

use crate::app::{UiTimeline, UiTimelineItem};

mod projector;
mod projectors;
mod shared;

pub use self::projector::{ProjectCtx, ToolProjector};
use self::projectors::edit::merge_edit_activity;
pub(crate) use self::projectors::edit::{patch_file_subject, patch_status_title};
use self::projectors::exploration::merge_exploration_activity;
use self::shared::activity_tool_name;

pub(crate) fn is_batch_tool_name(name: &str) -> bool {
    activity_tool_name(name) == "batch"
}

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
    Cancelled,
    Running,
    Partial,
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

fn merge_projected_activity(target: &mut ActivityBlock, incoming: ActivityBlock) -> bool {
    if target.call.kind == ActivityKind::Exploration
        && incoming.call.kind == ActivityKind::Exploration
        && target.result.status == ActivityStatus::Completed
        && incoming.result.status == ActivityStatus::Completed
    {
        return merge_exploration_activity(target, incoming);
    }
    if target.call.kind == ActivityKind::Edit
        && incoming.call.kind == ActivityKind::Edit
        && target.result.status == ActivityStatus::Completed
        && incoming.result.status == ActivityStatus::Completed
    {
        return merge_edit_activity(target, incoming);
    }
    false
}

#[derive(Clone)]
pub struct ActivityState {
    shell_handles: HashMap<String, String>,
    /// Tool-name -> projector registry, populated once at construction via
    /// `projectors::register_builtins`. Dispatch walks this map before
    /// falling through to the generic projector.
    registry: HashMap<&'static str, Arc<dyn ToolProjector>>,
    /// Live subagent host, plumbed through to projectors via `ProjectCtx`
    /// so they can fetch display-only metadata (capability, model, run
    /// state, per-completion stats) by agent name. `None` in test
    /// contexts that don't exercise subagent rendering.
    subagent_host: Option<Arc<dyn SubagentHost>>,
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
            subagent_host: None,
        };
        projectors::register_builtins(&mut state);
        state
    }

    /// Provide a live subagent host so subagent projectors can fetch
    /// display metadata (capability, model, run state) by agent name.
    /// Production code calls this once during App setup.
    pub fn set_subagent_host(&mut self, host: Arc<dyn SubagentHost>) {
        self.subagent_host = Some(host);
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

    pub fn project_tool_output(
        &mut self,
        name: &str,
        args: Value,
        output: lash_core::ToolCallOutput,
        duration_ms: u64,
    ) -> Vec<ActivityBlock> {
        let name = activity_tool_name(name);
        let result = match &output.outcome {
            lash_core::ToolCallOutcome::Failure(failure) => failure
                .raw
                .as_ref()
                .map(lash_core::ToolValue::to_json_value)
                .unwrap_or_else(|| output.value_for_projection()),
            lash_core::ToolCallOutcome::Cancelled(cancellation) => cancellation
                .raw
                .as_ref()
                .map(lash_core::ToolValue::to_json_value)
                .unwrap_or_else(|| output.value_for_projection()),
            lash_core::ToolCallOutcome::Success(_) => output.value_for_projection(),
        };
        let success = output.is_success();
        if name == "batch" {
            return self.blocks_for_batch_tool_call(&args, &result);
        }
        let status = match output.status() {
            lash_core::ToolCallStatus::Success => ActivityStatus::Completed,
            lash_core::ToolCallStatus::Failure => ActivityStatus::Failed,
            lash_core::ToolCallStatus::Cancelled => ActivityStatus::Cancelled,
        };
        let mut ctx = ProjectCtx {
            name,
            args,
            result,
            success,
            duration_ms,
            shell_handles: &mut self.shell_handles,
            subagent_host: self.subagent_host.as_ref(),
        };
        if status == ActivityStatus::Cancelled {
            return vec![projectors::generic::fallback_block(&mut ctx, status)];
        }
        if let Some(projector) = self.registry.get(name).cloned() {
            return projector.project(&mut ctx);
        }

        // Unregistered tool: fall back to the generic projector's
        // free function so we don't round-trip through the registry
        // (GenericProjector is already registered under `search_tools`).
        vec![projectors::generic::fallback_block(&mut ctx, status)]
    }

    pub(crate) fn append_tool_call_to_timeline(
        &mut self,
        timeline: &mut UiTimeline,
        name: &str,
        args: Value,
        output: lash_core::ToolCallOutput,
        duration_ms: u64,
    ) -> Vec<ActivityBlock> {
        let activities = self.project_tool_output(name, args, output, duration_ms);
        for activity in activities.iter().cloned() {
            Self::append_projected_activity_to_timeline(timeline, activity);
        }
        activities
    }

    pub(crate) fn append_projected_activity_to_timeline(
        timeline: &mut UiTimeline,
        activity: ActivityBlock,
    ) {
        if let Some(UiTimelineItem::Activity(existing)) = timeline.last_mut()
            && merge_projected_activity(existing, activity.clone())
        {
            return;
        }
        timeline.push(UiTimelineItem::Activity(Box::new(activity)));
    }

    fn blocks_for_batch_tool_call(&mut self, args: &Value, result: &Value) -> Vec<ActivityBlock> {
        let Some(entries) = batch_result_entries(result) else {
            return Vec::new();
        };
        let calls = batch_call_specs(args);

        entries
            .iter()
            .enumerate()
            .flat_map(|(index, item)| {
                let child_name = item
                    .get("tool")
                    .and_then(|value| value.as_str())
                    .or_else(|| calls.get(index).and_then(|call| call.tool_name.as_deref()))
                    .unwrap_or("tool")
                    .to_string();
                let child_args = calls
                    .get(index)
                    .map(|call| call.args.clone())
                    .unwrap_or_else(|| Value::Object(Default::default()));
                let child_success = item
                    .get("success")
                    .and_then(|value| value.as_bool())
                    .unwrap_or_else(|| item.get("error").is_none());
                let child_duration = item
                    .get("duration_ms")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0);
                let child_result = if let Some(result) = item.get("result") {
                    result.clone()
                } else if let Some(error) = item.get("error") {
                    error.clone()
                } else {
                    item.clone()
                };

                let child_output = if child_success {
                    lash_core::ToolCallOutput::success(child_result)
                } else {
                    *lash_core::ToolResult::err(child_result).output
                };

                self.project_tool_output(&child_name, child_args, child_output, child_duration)
            })
            .collect()
    }

    #[cfg(test)]
    pub fn project_tool_call(
        &mut self,
        name: &str,
        args: Value,
        result: Value,
        success: bool,
        duration_ms: u64,
    ) -> Vec<ActivityBlock> {
        let output = if success {
            lash_core::ToolCallOutput::success(result)
        } else {
            *lash_core::ToolResult::err(result).output
        };
        self.project_tool_output(name, args, output, duration_ms)
    }
}

#[derive(Clone, Debug, Default)]
struct BatchChildCallSpec {
    tool_name: Option<String>,
    args: Value,
}

fn batch_call_specs(args: &Value) -> Vec<BatchChildCallSpec> {
    if let Some(calls) = args.get("tool_calls").and_then(|value| value.as_array()) {
        return calls
            .iter()
            .map(|item| BatchChildCallSpec {
                tool_name: item
                    .get("tool")
                    .and_then(|value| value.as_str())
                    .map(str::to_string),
                args: item
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| Value::Object(Default::default())),
            })
            .collect();
    }
    if let Some(calls) = args.get("tool_uses").and_then(|value| value.as_array()) {
        return calls
            .iter()
            .map(|item| BatchChildCallSpec {
                tool_name: item
                    .get("recipient_name")
                    .and_then(|value| value.as_str())
                    .map(str::to_string),
                args: item
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| Value::Object(Default::default())),
            })
            .collect();
    }
    Vec::new()
}

fn batch_result_entries(result: &Value) -> Option<&Vec<Value>> {
    result
        .get("results")
        .and_then(|value| value.as_array())
        .or_else(|| result.as_array())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use async_trait::async_trait;
    use lash_subagents::{
        AgentMetadata, CloseAgentRequest, CloseAgentResponse, SpawnAgentRequest,
        SpawnAgentResponse, SubagentHost, WaitAgentRequest, WaitAgentResponse,
    };
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// In-memory `SubagentHost` for projector tests. Pre-populate with
    /// `insert(session_id, agent_name, AgentMetadata { … })` to feed the activity
    /// projector's profile/model/run_state lookups.
    #[derive(Default)]
    pub(crate) struct MockSubagentHost {
        metadata: Mutex<HashMap<String, AgentMetadata>>,
    }

    impl MockSubagentHost {
        pub(crate) fn insert(
            &self,
            session_id: impl Into<String>,
            agent_name: impl Into<String>,
            meta: AgentMetadata,
        ) {
            self.metadata
                .lock()
                .expect("metadata lock")
                .insert(format!("{}:{}", session_id.into(), agent_name.into()), meta);
        }
    }

    #[async_trait]
    impl SubagentHost for MockSubagentHost {
        fn agent_metadata(&self, session_id: &str, agent_name: &str) -> Option<AgentMetadata> {
            self.metadata
                .lock()
                .expect("metadata lock")
                .get(&format!("{session_id}:{agent_name}"))
                .cloned()
        }

        async fn spawn_agent(
            &self,
            _context: &lash_core::ToolContext,
            _request: SpawnAgentRequest,
        ) -> Result<SpawnAgentResponse, String> {
            unreachable!("MockSubagentHost is read-only")
        }

        async fn wait_agent(
            &self,
            _context: &lash_core::ToolContext,
            _request: WaitAgentRequest,
        ) -> Result<WaitAgentResponse, String> {
            unreachable!("MockSubagentHost is read-only")
        }

        async fn close_agent(
            &self,
            _context: &lash_core::ToolContext,
            _request: CloseAgentRequest,
        ) -> Result<CloseAgentResponse, String> {
            unreachable!("MockSubagentHost is read-only")
        }
    }

    pub(crate) fn explore_metadata(
        capability: impl Into<String>,
        model: impl Into<String>,
        variant: impl Into<String>,
    ) -> AgentMetadata {
        AgentMetadata {
            session_id: "session".to_string(),
            parent_session_id: Some("root".to_string()),
            capability: Some(capability.into()),
            state: "running".to_string(),
            model: model.into(),
            model_variant: Some(variant.into()),
            last_iterations: None,
            last_tool_calls: None,
            last_token_usage: None,
        }
    }

    #[test]
    fn batch_expands_into_child_tool_blocks() {
        let mut state = ActivityState::default();
        let blocks = state.project_tool_call(
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
    fn typed_cancelled_output_projects_cancelled_activity_status() {
        let mut state = ActivityState::default();
        let blocks = state.project_tool_output(
            "read_file",
            json!({ "path": "README.md" }),
            lash_core::ToolCallOutput::cancelled(lash_core::ToolCancellation::runtime(
                "tool call cancelled",
            )),
            1,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].result.status, ActivityStatus::Cancelled);
    }

    #[test]
    fn batch_normalizes_namespaced_child_tool_blocks() {
        let mut state = ActivityState::default();
        let blocks = state.project_tool_call(
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
        let blocks = state.project_tool_call(
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
        let blocks = state.project_tool_call(
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
    fn tool_use_batch_results_expand_into_child_tool_blocks() {
        let mut state = ActivityState::default();
        let blocks = state.project_tool_call(
            "batch",
            json!({
                "tool_uses": [
                    {
                        "recipient_name": "functions.read_file",
                        "parameters": {"path": "README.md"}
                    },
                    {
                        "recipient_name": "functions.grep",
                        "parameters": {"query": "OpenAI", "path": "README.md"}
                    }
                ]
            }),
            json!([
                {"success": true, "result": "README body", "duration_ms": 8},
                {"success": true, "result": "match", "duration_ms": 13}
            ]),
            true,
            21,
        );

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].call.tool_name, "read_file");
        assert_eq!(blocks[1].call.tool_name, "grep");
    }

    #[test]
    fn write_stdin_uses_session_id_argument() {
        let mut state = ActivityState::default();
        state.shell_handles.insert("7".into(), "python3 -q".into());
        let blocks = state.project_tool_call(
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
        state.project_tool_call(
            "start_command",
            json!({"cmd":"python3 -q"}),
            json!({
                "session_id": 7,
                "output": ""
            }),
            true,
            1,
        );

        let wrote = state.project_tool_call(
            "write_stdin",
            json!({"session_id":7,"chars":"print(2 + 2)\n","poll_ms":1000}),
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
        state.project_tool_call(
            "start_command",
            json!({"cmd":"python3 -q"}),
            json!({
                "session_id": 7,
                "output": ""
            }),
            true,
            1,
        );

        let polled = state.project_tool_call(
            "write_stdin",
            json!({"session_id":7,"chars":"","poll_ms":1000}),
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
        state.project_tool_call(
            "start_command",
            json!({"cmd":"python3 -q"}),
            json!({
                "session_id": 7,
                "output": ""
            }),
            true,
            1,
        );

        let polled = state.project_tool_call(
            "write_stdin",
            json!({"session_id":7,"chars":"","poll_ms":1000}),
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
        let search_blocks = state.project_tool_call(
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
        let ask_blocks = state.project_tool_call(
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
        let ask_blocks = state.project_tool_call(
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
        let blocks = state.project_tool_call(
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
        let host = Arc::new(MockSubagentHost::default());
        host.insert(
            "root",
            "probe_repo_shape",
            explore_metadata("explore", "gpt-5.4-mini", "low"),
        );
        let mut state = ActivityState::default();
        state.set_subagent_host(host);
        let blocks = state.project_tool_call(
            "spawn_agent",
            json!({
                "agent_name":"probe_repo_shape",
                "task":"In /home/sam/code/lash, inspect the repo shape only. Reply with the top-level summary.",
                "capability":"explore"
            }),
            json!({
                "agent_name":"probe_repo_shape",
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
                "Agent probe_repo_shape".to_string(),
                "Profile explore capability".to_string(),
            ]
        );
    }
}
