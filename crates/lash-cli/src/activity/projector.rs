use std::collections::HashMap;

use serde_json::Value;

use super::ActivityBlock;

/// Interprets a tool call into one or more `ActivityBlock`s. One
/// `ToolProjector` per tool family (or per individual tool). Registered
/// in `ActivityState::new` via `projectors::register_builtins`.
///
/// This is layer 1 of the two-layer render pipeline — it maps raw JSON
/// tool calls onto structured display descriptors. Layer 2 (the
/// per-variant renderer in `render/activity.rs`) then turns those
/// descriptors into terminal lines. Keeping the layers separate is
/// what lets the block render cache stay keyed on stable
/// `ActivityBlock` data across expand-level transitions.
pub trait ToolProjector: Send + Sync {
    /// The normalized tool names (after `activity_tool_name` stripping)
    /// this projector handles. Returned once at registration time, not
    /// called on the hot path.
    fn tool_names(&self) -> &'static [&'static str];

    /// Project a tool call. Return an empty vec to suppress the call
    /// from display (used by the shell write_stdin poll-suppress
    /// branch). Return multiple blocks when a single tool call expands
    /// into several display blocks (batch expansion goes through the
    /// dispatcher, not a projector).
    fn project(&self, ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock>;
}

/// Mutable context passed to every `ToolProjector::project` call.
///
/// `args` and `result` are owned so projectors can move them into the
/// `ActivityBlock` they construct (no clone on the hot path). The
/// stateful shell handle map is split out of `ActivityState` as a
/// disjoint mutable borrow so projectors can mutate it without holding
/// a reference to the whole state.
pub struct ProjectCtx<'a> {
    /// Normalized tool name (after `activity_tool_name` prefix stripping).
    pub name: &'a str,
    pub args: Value,
    pub result: Value,
    pub success: bool,
    pub duration_ms: u64,
    /// Shell session handles: session_id → original command. Used by
    /// `exec_command` (insert on running start) and `write_stdin`
    /// (remove on exit, or_insert if still running).
    pub shell_handles: &'a mut HashMap<String, String>,
}
