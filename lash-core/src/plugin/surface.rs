use lash_sansio::ToolCallOutput;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use super::*;

#[derive(Clone)]
pub struct ToolSurfaceContext {
    pub session_id: String,
    pub mode: ExecutionMode,
    pub tools: Vec<ToolManifest>,
    pub resolve_contract: Option<lash_sansio::ToolContractResolver>,
    pub tool_access: SessionToolAccess,
    pub subagent: Option<SubagentSessionAuthority>,
}

#[derive(Clone, Debug)]
pub struct ToolDiscoveryContext {
    pub session_id: String,
    pub mode: ExecutionMode,
    pub catalog: Vec<serde_json::Value>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ToolDiscoveryContribution {
    pub tools: Vec<ToolDiscoveryToolContribution>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ToolDiscoveryToolContribution {
    pub tool_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct PluginAbort {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, Default)]
pub struct TurnPreparation {
    pub messages: crate::MessageSequence,
    pub events: Vec<crate::SessionEvent>,
    pub abort: Option<PluginAbort>,
}

#[derive(Clone)]
pub struct PrepareTurnRequest {
    pub session_id: String,
    pub state: SessionReadView,
    pub messages: crate::MessageSequence,
    pub host: Arc<dyn TurnHookHost>,
    pub turn_context: crate::TurnContext,
}

#[derive(Clone, Debug, Default)]
pub struct CheckpointApplication {
    pub messages: Vec<PluginMessage>,
    pub events: Vec<crate::SessionEvent>,
    pub abort: Option<PluginAbort>,
}

#[derive(Clone, Debug)]
pub struct TurnFinalization {
    pub turn: AssembledTurn,
    pub events: Vec<crate::SessionEvent>,
}

pub(crate) async fn emit_plugin_surface_events(
    event_tx: &mpsc::Sender<crate::SessionEvent>,
    plugin_id: &str,
    events: Vec<PluginSurfaceEvent>,
) {
    for event in plugin_surface_session_events(plugin_id, events) {
        crate::session_model::send_event(event_tx, event).await;
    }
}

pub(crate) fn plugin_surface_session_events(
    plugin_id: &str,
    events: Vec<PluginSurfaceEvent>,
) -> Vec<crate::SessionEvent> {
    events
        .into_iter()
        .map(|event| crate::SessionEvent::PluginEvent {
            plugin_id: plugin_id.to_string(),
            event,
        })
        .collect()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)]
pub enum PluginDirective {
    AbortTurn {
        code: String,
        message: String,
    },
    EnqueueMessages {
        messages: Vec<PluginMessage>,
    },
    CreateSession {
        request: Box<SessionCreateRequest>,
    },
    HandoffSession {
        session_id: String,
    },
    ReplaceToolArgs {
        args: serde_json::Value,
    },
    ShortCircuitTool {
        output: ToolCallOutput,
    },
    EmitEvents {
        events: Vec<PluginSurfaceEvent>,
    },
    EmitTrace {
        name: String,
        #[serde(default)]
        payload: serde_json::Value,
        #[serde(default)]
        context: Box<lash_trace::TraceContext>,
    },
}

impl PluginDirective {
    pub fn short_circuit(result: ToolResult) -> Self {
        Self::ShortCircuitTool {
            output: *result.output,
        }
    }

    pub fn into_tool_result(self) -> Option<ToolResult> {
        match self {
            Self::ShortCircuitTool { output } => Some(ToolResult::from_output(output)),
            _ => None,
        }
    }

    pub fn emit_events(events: Vec<PluginSurfaceEvent>) -> Self {
        Self::EmitEvents { events }
    }

    pub fn emit_trace(name: impl Into<String>, payload: serde_json::Value) -> Self {
        Self::EmitTrace {
            name: name.into(),
            payload,
            context: Box::new(lash_trace::TraceContext::default()),
        }
    }
}
