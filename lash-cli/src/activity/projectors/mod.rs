//! Per-tool projectors.
//!
//! Each module in this directory implements [`ToolProjector`] for one or
//! more related tool names. The dispatcher in `activity::ActivityState`
//! looks up a projector by tool name on every tool call and delegates
//! block construction. A `GenericProjector` catches any name that
//! doesn't have a dedicated projector.
//!
//! This is layer 1 of the two-layer render pipeline — see
//! [`crate::activity::projector`] for the trait definition and the
//! shape of the rendering boundary.

use super::ActivityState;

pub(crate) mod ask;
pub(crate) mod edit;
pub(crate) mod exploration;
pub(crate) mod generic;
pub(crate) mod lashlang;
pub(crate) mod shell;
pub(crate) mod snippet;
pub(crate) mod subagents;
pub(crate) mod web;

/// Register every built-in projector with the given `ActivityState`.
/// Called from `ActivityState::new`. The generic projector is also
/// registered under its primary tool name (`search_tools`), and
/// additionally used as the fallback path in
/// `ActivityState::blocks_for_tool_call` for names with no dedicated
/// projector.
pub(super) fn register_builtins(state: &mut ActivityState) {
    state.register(ask::AskProjector);
    state.register(edit::EditProjector);
    state.register(exploration::ExplorationProjector);
    state.register(generic::GenericProjector);
    state.register(lashlang::LashlangProjector);
    state.register(shell::ShellProjector);
    state.register(snippet::SnippetProjector);
    state.register(subagents::SubagentProjector);
    state.register(web::WebProjector);
}
