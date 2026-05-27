use std::collections::BTreeSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionHandle {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    pub policy: SessionPolicy,
}

pub type SessionSnapshot = RuntimeSessionState;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionStartPoint {
    Empty,
    CurrentSession,
    ExistingSession { session_id: String },
    Snapshot { snapshot: Box<SessionSnapshot> },
}

#[derive(Clone, Debug)]
pub struct PluginOwned<T> {
    pub plugin_id: String,
    pub value: T,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionPluginSource {
    CurrentHostFresh,
    #[default]
    CurrentSessionFork,
}

#[derive(Clone)]
pub struct SessionContextSurface {
    pub include_base_tools: bool,
    pub tool_providers: Vec<Arc<dyn ToolProvider>>,
    pub prompt_contributions: Vec<PromptContribution>,
}

impl Default for SessionContextSurface {
    fn default() -> Self {
        Self {
            include_base_tools: true,
            tool_providers: Vec::new(),
            prompt_contributions: Vec::new(),
        }
    }
}

impl std::fmt::Debug for SessionContextSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionContextSurface")
            .field("include_base_tools", &self.include_base_tools)
            .field("tool_provider_count", &self.tool_providers.len())
            .field(
                "prompt_contribution_count",
                &self.prompt_contributions.len(),
            )
            .finish()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionRelation {
    #[default]
    Root,
    Child {
        parent_session_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        originating_tool_call_id: Option<String>,
    },
    Handoff {
        parent_session_id: String,
        reason: String,
        #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
        metadata: serde_json::Map<String, serde_json::Value>,
    },
}

impl SessionRelation {
    pub fn parent_session_id(&self) -> Option<&str> {
        match self {
            Self::Root => None,
            Self::Child {
                parent_session_id, ..
            }
            | Self::Handoff {
                parent_session_id, ..
            } => Some(parent_session_id),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionCreateRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default)]
    pub relation: SessionRelation,
    pub start: SessionStartPoint,
    #[serde(default)]
    pub policy: Option<SessionPolicy>,
    #[serde(default)]
    pub plugin_source: SessionPluginSource,
    #[serde(default)]
    pub initial_nodes: Vec<SessionAppendNode>,
    /// Optional seed message dispatched as the new session's first turn
    /// input. The runtime stashes it during `create_session`; any host
    /// that drives turns on the new session can claim it via
    /// `RuntimeSessionHost::take_first_turn_input`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_turn_input: Option<PluginMessage>,
    #[serde(default)]
    pub tool_access: SessionToolAccess,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent: Option<SubagentSessionContext>,
    #[serde(skip)]
    pub context_surface: SessionContextSurface,
    /// Plugin-owned options that configure plugin behavior at session
    /// creation time. Each plugin decodes only the entry keyed by its id.
    #[serde(default)]
    pub plugin_options: PluginOptions,
    /// Label for the token-cost ledger. When this session's turns
    /// complete, their token usage is accumulated under this label on
    /// the parent session's `token_ledger`. Examples: `"subagent"`,
    /// `"compaction"`. Defaults to `"child"` if unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage_source: Option<String>,
}

impl SessionCreateRequest {
    pub fn root(start: SessionStartPoint, plugin_options: PluginOptions) -> Self {
        Self {
            session_id: Some(uuid::Uuid::new_v4().to_string()),
            relation: SessionRelation::Root,
            start,
            policy: None,
            plugin_source: SessionPluginSource::CurrentHostFresh,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: SessionToolAccess::default(),
            subagent: None,
            context_surface: SessionContextSurface::default(),
            plugin_options,
            usage_source: None,
        }
    }

    pub fn root_with_policy(
        start: SessionStartPoint,
        policy: SessionPolicy,
        plugin_options: PluginOptions,
    ) -> Self {
        Self {
            policy: Some(policy),
            ..Self::root(start, plugin_options)
        }
    }

    pub fn child_session(
        parent_session_id: impl Into<String>,
        start: SessionStartPoint,
        plugin_options: PluginOptions,
    ) -> Self {
        Self {
            session_id: Some(uuid::Uuid::new_v4().to_string()),
            relation: SessionRelation::Child {
                parent_session_id: parent_session_id.into(),
                originating_tool_call_id: None,
            },
            start,
            policy: None,
            plugin_source: SessionPluginSource::CurrentHostFresh,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: SessionToolAccess::default(),
            subagent: None,
            context_surface: SessionContextSurface::default(),
            plugin_options,
            usage_source: None,
        }
    }

    pub fn child_session_with_policy(
        parent_session_id: impl Into<String>,
        start: SessionStartPoint,
        policy: SessionPolicy,
        plugin_options: PluginOptions,
    ) -> Self {
        Self {
            policy: Some(policy),
            ..Self::child_session(parent_session_id, start, plugin_options)
        }
    }

    pub fn handoff_session(
        parent_session_id: impl Into<String>,
        reason: impl Into<String>,
        metadata: serde_json::Map<String, serde_json::Value>,
        plugin_options: PluginOptions,
    ) -> Self {
        Self {
            session_id: Some(uuid::Uuid::new_v4().to_string()),
            relation: SessionRelation::Handoff {
                parent_session_id: parent_session_id.into(),
                reason: reason.into(),
                metadata,
            },
            start: SessionStartPoint::Empty,
            policy: None,
            plugin_source: SessionPluginSource::CurrentHostFresh,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: SessionToolAccess::default(),
            subagent: None,
            context_surface: SessionContextSurface::default(),
            plugin_options,
            usage_source: None,
        }
    }

    pub fn child(
        parent_session_id: impl Into<String>,
        start: SessionStartPoint,
        policy: SessionPolicy,
        plugin_options: PluginOptions,
        usage_source: impl Into<String>,
    ) -> Self {
        Self::related(
            SessionRelation::Child {
                parent_session_id: parent_session_id.into(),
                originating_tool_call_id: None,
            },
            start,
            Some(policy),
            plugin_options,
            usage_source,
        )
    }

    pub fn child_inheriting_policy(
        parent_session_id: impl Into<String>,
        start: SessionStartPoint,
        plugin_options: PluginOptions,
        usage_source: impl Into<String>,
    ) -> Self {
        Self::related(
            SessionRelation::Child {
                parent_session_id: parent_session_id.into(),
                originating_tool_call_id: None,
            },
            start,
            None,
            plugin_options,
            usage_source,
        )
    }

    pub fn handoff(
        parent_session_id: impl Into<String>,
        policy: SessionPolicy,
        plugin_options: PluginOptions,
        reason: impl Into<String>,
        metadata: serde_json::Map<String, serde_json::Value>,
        usage_source: impl Into<String>,
    ) -> Self {
        Self::related(
            SessionRelation::Handoff {
                parent_session_id: parent_session_id.into(),
                reason: reason.into(),
                metadata,
            },
            SessionStartPoint::Empty,
            Some(policy),
            plugin_options,
            usage_source,
        )
    }

    fn related(
        relation: SessionRelation,
        start: SessionStartPoint,
        policy: Option<SessionPolicy>,
        plugin_options: PluginOptions,
        usage_source: impl Into<String>,
    ) -> Self {
        Self {
            session_id: Some(uuid::Uuid::new_v4().to_string()),
            relation,
            start,
            policy,
            plugin_source: SessionPluginSource::CurrentHostFresh,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: SessionToolAccess::default(),
            subagent: None,
            context_surface: SessionContextSurface::default(),
            plugin_options,
            usage_source: Some(usage_source.into()),
        }
    }

    pub fn with_plugin_source(mut self, plugin_source: SessionPluginSource) -> Self {
        self.plugin_source = plugin_source;
        self
    }

    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_initial_nodes(mut self, initial_nodes: Vec<SessionAppendNode>) -> Self {
        self.initial_nodes = initial_nodes;
        self
    }

    pub fn with_first_turn_input(mut self, first_turn_input: PluginMessage) -> Self {
        self.first_turn_input = Some(first_turn_input);
        self
    }

    pub fn with_tool_access(mut self, tool_access: SessionToolAccess) -> Self {
        self.tool_access = tool_access;
        self
    }

    pub fn with_subagent_context(mut self, subagent: SubagentSessionContext) -> Self {
        self.subagent = Some(subagent);
        self
    }

    pub fn with_originating_tool_call_id(mut self, tool_call_id: impl Into<String>) -> Self {
        if let SessionRelation::Child {
            originating_tool_call_id,
            ..
        } = &mut self.relation
        {
            *originating_tool_call_id = Some(tool_call_id.into());
        }
        self
    }

    pub fn with_context_surface(mut self, context_surface: SessionContextSurface) -> Self {
        self.context_surface = context_surface;
        self
    }

    pub fn with_usage_source(mut self, usage_source: impl Into<String>) -> Self {
        self.usage_source = Some(usage_source.into());
        self
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SessionToolAccess {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub hidden_tools: BTreeSet<String>,
}

impl SessionToolAccess {
    pub fn hides(&self, name: &str) -> bool {
        self.hidden_tools.contains(name)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubagentSessionContext {
    pub parent_session_id: String,
    pub capability: String,
    pub depth: u8,
    pub max_depth: u8,
}

/// Plugin-owned payloads carried on a `SessionCreateRequest`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)]
pub enum SessionAppendNode {
    Message {
        message: PluginMessage,
    },
    ProtocolEvent {
        event: crate::ProtocolEvent,
    },
    Plugin {
        plugin_type: String,
        #[serde(default)]
        body: serde_json::Value,
    },
}

impl SessionAppendNode {
    pub fn message(message: PluginMessage) -> Self {
        Self::Message { message }
    }

    pub fn plugin(plugin_type: impl Into<String>, body: serde_json::Value) -> Self {
        Self::Plugin {
            plugin_type: plugin_type.into(),
            body,
        }
    }

    pub fn protocol_event(event: crate::ProtocolEvent) -> Self {
        Self::ProtocolEvent { event }
    }
}
