//! The `SubagentHost` trait — the abstract contract every subagent
//! host implementation satisfies.
//!
//! This file is intentionally thin: it holds the trait and re-exports
//! the API-facing types and the primary [`LocalSubagentHost`]
//! implementation so historical `host::Type` paths keep working after
//! the module was split. Look elsewhere for behavior:
//!
//!   * [`crate::types`]   — request/response/event data types
//!   * [`crate::routing`] — pure path + event-matching helpers, plus
//!     [`truncate_snapshot_to_recent_turns`]
//!   * [`crate::queue`]   — internal state primitives (AgentRecord,
//!     AgentTree, HostState, QueuedTurn, …) and per-completion event
//!     builders, including the `queue_event` purge-stale invariant
//!   * [`crate::local`]   — the in-process [`LocalSubagentHost`] that
//!     implements this trait, including `force_close_subtree` and all
//!     turn-lifecycle handling

use async_trait::async_trait;

pub use crate::local::LocalSubagentHost;
pub use crate::routing::truncate_snapshot_to_recent_turns;
pub use crate::types::{
    AgentMetadata, CloseAgentRequest, CloseAgentResponse, SpawnAgentRequest, SpawnAgentResponse,
    WaitAgentClosed, WaitAgentCompletion, WaitAgentEvent, WaitAgentRequest, WaitAgentResponse,
    WaitUntil,
};

#[async_trait]
pub trait SubagentHost: Send + Sync {
    /// Return display-only metadata about a direct child agent.
    ///
    /// UI consumers (the activity projector) call this to fetch model name,
    /// capability, run state, and per-completion stats for the dock without
    /// needing those fields to live on the model-facing wire shapes.
    /// Returns `None` if no direct child exists with `agent_name`.
    fn agent_metadata(&self, session_id: &str, agent_name: &str) -> Option<AgentMetadata> {
        let _ = (session_id, agent_name);
        None
    }

    async fn spawn_agent(
        &self,
        context: &lash::ToolContext,
        request: SpawnAgentRequest,
    ) -> Result<SpawnAgentResponse, String>;

    async fn wait_agent(
        &self,
        context: &lash::ToolContext,
        request: WaitAgentRequest,
    ) -> Result<WaitAgentResponse, String>;

    async fn close_agent(
        &self,
        context: &lash::ToolContext,
        request: CloseAgentRequest,
    ) -> Result<CloseAgentResponse, String>;
}
