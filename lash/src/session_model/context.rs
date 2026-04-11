//! Context-preparation vocabulary shared between the runtime and the
//! plugin host. The concrete strategy (rolling history, overflow
//! recovery, `/compact`) lives in
//! [`crate::plugin_builtin::rolling_history`] and is dispatched through
//! the [`TurnContextTransform`](crate::plugin::TurnContextTransform) and
//! [`HistoryRewriter`](crate::plugin::HistoryRewriter) plugin capability
//! surfaces.

use std::sync::Arc;

use crate::plugin::PromptContribution;
use crate::{Message, ToolProvider};

/// Output of the per-turn context transform pipeline — the messages,
/// prompt contributions, and tool providers the runtime hands to the
/// LLM call.
#[derive(Clone)]
pub struct PreparedContext {
    pub messages: Vec<Message>,
    pub prompt_contributions: Vec<PromptContribution>,
    pub tool_providers: Vec<Arc<dyn ToolProvider>>,
    pub include_base_tools: bool,
}

impl Default for PreparedContext {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            prompt_contributions: Vec::new(),
            tool_providers: Vec::new(),
            include_base_tools: true,
        }
    }
}
