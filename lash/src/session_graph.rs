use std::collections::{HashMap, HashSet};

use base64::Engine;
use chrono::Utc;
use sha2::Digest;

use crate::{
    DynamicStateSnapshot, ExecutionMode, Message, MessageRole, PluginSessionSnapshot, PromptUsage,
    TokenUsage, ToolCallRecord,
};

pub const INTERNAL_TOOL_CALL_PLUGIN_TYPE: &str = "lash.tool_call_record";
pub const INTERNAL_SESSION_CONFIG_PLUGIN_TYPE: &str = "lash.session_config";
pub const INTERNAL_TURN_STATE_PLUGIN_TYPE: &str = "lash.turn_state";
pub const INTERNAL_DYNAMIC_STATE_PLUGIN_TYPE: &str = "lash.dynamic_state";
pub const INTERNAL_PLUGIN_SNAPSHOT_PLUGIN_TYPE: &str = "lash.plugin_snapshot";
pub const INTERNAL_EXECUTION_STATE_PLUGIN_TYPE: &str = "lash.execution_state";

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SessionGraph {
    #[serde(default)]
    pub nodes: Vec<SessionNodeRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leaf_node_id: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SessionNodeRecord {
    pub node_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_node_id: Option<String>,
    pub timestamp: String,
    #[serde(flatten)]
    pub payload: SessionNodePayload,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionNodePayload {
    Message {
        message: Message,
    },
    Plugin {
        plugin_type: String,
        body: serde_json::Value,
    },
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolCallPluginBody {
    pub stable_key: String,
    pub record: ToolCallRecord,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PersistedSessionConfig {
    pub provider_id: String,
    pub configured_model: String,
    pub context_window: u64,
    pub execution_mode: ExecutionMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_variant: Option<String>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct PersistedTurnState {
    pub iteration: usize,
    #[serde(default)]
    pub token_usage: TokenUsage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_prompt_usage: Option<PromptUsage>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ExecutionStatePluginBody {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_base64: Option<String>,
}

impl ExecutionStatePluginBody {
    pub fn from_snapshot(snapshot: Option<&[u8]>) -> Self {
        Self {
            snapshot_base64: snapshot
                .map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes)),
        }
    }

    pub fn snapshot_bytes(&self) -> Option<Vec<u8>> {
        self.snapshot_base64.as_ref().and_then(|encoded| {
            base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .ok()
        })
    }
}

#[derive(Clone, Debug)]
pub struct SessionMessageTreeNode {
    pub node_id: String,
    pub parent_message_node_id: Option<String>,
    pub message: Message,
    pub timestamp: String,
    pub children: Vec<SessionMessageTreeNode>,
    pub active: bool,
}

#[derive(Clone)]
enum ProjectionItem<'a> {
    Message(&'a Message),
    ToolCall {
        stable_key: String,
        record: &'a ToolCallRecord,
    },
}

impl SessionNodeRecord {
    pub fn message(&self) -> Option<&Message> {
        match &self.payload {
            SessionNodePayload::Message { message } => Some(message),
            SessionNodePayload::Plugin { .. } => None,
        }
    }

    pub fn plugin(&self) -> Option<(&str, &serde_json::Value)> {
        match &self.payload {
            SessionNodePayload::Message { .. } => None,
            SessionNodePayload::Plugin { plugin_type, body } => Some((plugin_type.as_str(), body)),
        }
    }
}

impl SessionGraph {
    pub fn append_message(&mut self, mut message: Message) -> String {
        if message.id.is_empty() {
            message.id = fresh_node_id("m");
        }
        let node_id = message.id.clone();
        self.nodes.push(SessionNodeRecord {
            node_id: node_id.clone(),
            parent_node_id: self.leaf_node_id.clone(),
            timestamp: Utc::now().to_rfc3339(),
            payload: SessionNodePayload::Message { message },
        });
        self.leaf_node_id = Some(node_id.clone());
        node_id
    }

    pub fn append_plugin(
        &mut self,
        plugin_type: impl Into<String>,
        body: serde_json::Value,
    ) -> String {
        let node_id = fresh_node_id("x");
        self.nodes.push(SessionNodeRecord {
            node_id: node_id.clone(),
            parent_node_id: self.leaf_node_id.clone(),
            timestamp: Utc::now().to_rfc3339(),
            payload: SessionNodePayload::Plugin {
                plugin_type: plugin_type.into(),
                body,
            },
        });
        self.leaf_node_id = Some(node_id.clone());
        node_id
    }

    pub fn active_path_nodes(&self) -> Vec<&SessionNodeRecord> {
        let by_id = self
            .nodes
            .iter()
            .map(|node| (node.node_id.as_str(), node))
            .collect::<HashMap<_, _>>();
        let mut out = Vec::new();
        let mut current = self
            .leaf_node_id
            .as_deref()
            .and_then(|node_id| by_id.get(node_id).copied());
        while let Some(node) = current {
            out.push(node);
            current = node
                .parent_node_id
                .as_deref()
                .and_then(|node_id| by_id.get(node_id).copied());
        }
        out.reverse();
        out
    }

    pub fn project_messages(&self) -> Vec<Message> {
        self.active_path_nodes()
            .into_iter()
            .filter_map(|node| node.message().cloned())
            .collect()
    }

    pub fn project_tool_calls(&self) -> Vec<ToolCallRecord> {
        self.active_path_nodes()
            .into_iter()
            .filter_map(|node| match node.plugin() {
                Some((INTERNAL_TOOL_CALL_PLUGIN_TYPE, body)) => {
                    serde_json::from_value::<ToolCallPluginBody>(body.clone())
                        .ok()
                        .map(|body| body.record)
                }
                _ => None,
            })
            .collect()
    }

    pub fn active_path_plugins(&self, plugin_type: &str) -> Vec<&serde_json::Value> {
        self.active_path_nodes()
            .into_iter()
            .filter_map(|node| match node.plugin() {
                Some((kind, body)) if kind == plugin_type => Some(body),
                _ => None,
            })
            .collect()
    }

    pub fn latest_plugin(&self, plugin_type: &str) -> Option<&serde_json::Value> {
        self.active_path_nodes()
            .into_iter()
            .rev()
            .find_map(|node| match node.plugin() {
                Some((kind, body)) if kind == plugin_type => Some(body),
                _ => None,
            })
    }

    pub fn latest_plugin_state<T>(&self, plugin_type: &str) -> Option<T>
    where
        T: serde::de::DeserializeOwned,
    {
        self.latest_plugin(plugin_type)
            .and_then(|body| serde_json::from_value(body.clone()).ok())
    }

    pub fn set_plugin_state<T>(&mut self, plugin_type: &str, state: &T) -> Option<String>
    where
        T: serde::Serialize,
    {
        let body = serde_json::to_value(state).ok()?;
        if self.latest_plugin(plugin_type) == Some(&body) {
            return None;
        }
        Some(self.append_plugin(plugin_type.to_string(), body))
    }

    pub fn latest_session_config(&self) -> Option<PersistedSessionConfig> {
        self.latest_plugin_state(INTERNAL_SESSION_CONFIG_PLUGIN_TYPE)
    }

    pub fn latest_turn_state(&self) -> Option<PersistedTurnState> {
        self.latest_plugin_state(INTERNAL_TURN_STATE_PLUGIN_TYPE)
    }

    pub fn latest_dynamic_state(&self) -> Option<DynamicStateSnapshot> {
        self.latest_plugin_state(INTERNAL_DYNAMIC_STATE_PLUGIN_TYPE)
    }

    pub fn latest_plugin_snapshot(&self) -> Option<PluginSessionSnapshot> {
        self.latest_plugin_state(INTERNAL_PLUGIN_SNAPSHOT_PLUGIN_TYPE)
    }

    pub fn latest_execution_state(&self) -> Option<Option<Vec<u8>>> {
        self.latest_plugin_state::<ExecutionStatePluginBody>(INTERNAL_EXECUTION_STATE_PLUGIN_TYPE)
            .map(|body| body.snapshot_bytes())
    }

    pub fn record_runtime_state(
        &mut self,
        config: &PersistedSessionConfig,
        turn_state: &PersistedTurnState,
        dynamic_state: Option<&DynamicStateSnapshot>,
        plugin_snapshot: Option<&PluginSessionSnapshot>,
        execution_state_snapshot: Option<&[u8]>,
    ) {
        let _ = self.set_plugin_state(INTERNAL_SESSION_CONFIG_PLUGIN_TYPE, config);
        let _ = self.set_plugin_state(INTERNAL_TURN_STATE_PLUGIN_TYPE, turn_state);
        if let Some(dynamic_state) = dynamic_state {
            let _ = self.set_plugin_state(INTERNAL_DYNAMIC_STATE_PLUGIN_TYPE, dynamic_state);
        }
        if let Some(plugin_snapshot) = plugin_snapshot {
            let _ = self.set_plugin_state(INTERNAL_PLUGIN_SNAPSHOT_PLUGIN_TYPE, plugin_snapshot);
        }
        let _ = self.set_plugin_state(
            INTERNAL_EXECUTION_STATE_PLUGIN_TYPE,
            &ExecutionStatePluginBody::from_snapshot(execution_state_snapshot),
        );
    }

    pub fn user_message_count(&self) -> usize {
        self.nodes
            .iter()
            .filter_map(SessionNodeRecord::message)
            .filter(|message| matches!(message.role, MessageRole::User))
            .count()
    }

    pub fn first_user_message(&self) -> String {
        self.nodes
            .iter()
            .filter_map(SessionNodeRecord::message)
            .find(|message| matches!(message.role, MessageRole::User))
            .map(first_message_search_text)
            .unwrap_or_default()
    }

    pub fn branch_to(&mut self, node_id: Option<String>) {
        self.leaf_node_id = node_id;
    }

    /// If `leaf_node_id` points to a node that no longer exists in
    /// `self.nodes` (e.g. after compaction rewrote the graph, or a
    /// stored session referenced a node that was later purged), fall
    /// back to the most recent message node. Returns `true` if the
    /// leaf was repaired. Call this on load paths where an orphan
    /// leaf would project to an empty transcript and silently drop
    /// the user's history.
    pub fn heal_orphaned_leaf(&mut self) -> bool {
        if let Some(leaf) = self.leaf_node_id.as_ref()
            && self.find_node(leaf).is_none()
        {
            let fallback = self
                .nodes
                .iter()
                .rev()
                .find(|node| node.message().is_some())
                .map(|node| node.node_id.clone());
            self.leaf_node_id = fallback;
            return true;
        }
        false
    }

    pub fn fork_current_path(&self) -> SessionGraph {
        let path = self.active_path_nodes();
        SessionGraph {
            nodes: path.into_iter().cloned().collect(),
            leaf_node_id: self.leaf_node_id.clone(),
        }
    }

    pub fn find_node(&self, node_id: &str) -> Option<&SessionNodeRecord> {
        self.nodes.iter().find(|node| node.node_id == node_id)
    }

    pub fn merge_active_projection(&mut self, messages: &[Message], tool_calls: &[ToolCallRecord]) {
        let current_nodes = self.active_path_nodes();
        let target = build_projection_items(messages, tool_calls);

        let mut preserved_ids = Vec::new();
        let mut target_idx = 0usize;
        for node in current_nodes {
            if let Some(key) = recognized_projection_key(node) {
                let Some(target_item) = target.get(target_idx) else {
                    break;
                };
                if key != target_item_key(target_item) {
                    break;
                }
                preserved_ids.push(node.node_id.clone());
                target_idx += 1;
            } else {
                preserved_ids.push(node.node_id.clone());
            }
        }

        self.leaf_node_id = preserved_ids.last().cloned();
        let mut existing_ids = self
            .nodes
            .iter()
            .map(|node| node.node_id.clone())
            .collect::<HashSet<_>>();

        for item in target.into_iter().skip(target_idx) {
            let parent_node_id = self.leaf_node_id.clone();
            let node = match item {
                ProjectionItem::Message(message) => SessionNodeRecord {
                    node_id: message.id.clone(),
                    parent_node_id,
                    timestamp: Utc::now().to_rfc3339(),
                    payload: SessionNodePayload::Message {
                        message: message.clone(),
                    },
                },
                ProjectionItem::ToolCall { stable_key, record } => {
                    let node_id = unique_plugin_node_id(&stable_key, &existing_ids);
                    SessionNodeRecord {
                        node_id,
                        parent_node_id,
                        timestamp: Utc::now().to_rfc3339(),
                        payload: SessionNodePayload::Plugin {
                            plugin_type: INTERNAL_TOOL_CALL_PLUGIN_TYPE.to_string(),
                            body: serde_json::to_value(ToolCallPluginBody {
                                stable_key,
                                record: record.clone(),
                            })
                            .unwrap_or(serde_json::Value::Null),
                        },
                    }
                }
            };
            existing_ids.insert(node.node_id.clone());
            self.leaf_node_id = Some(node.node_id.clone());
            self.nodes.push(node);
        }
    }

    pub fn from_projection(messages: &[Message], tool_calls: &[ToolCallRecord]) -> Self {
        let mut graph = Self::default();
        graph.merge_active_projection(messages, tool_calls);
        graph
    }

    pub fn message_tree(&self) -> Vec<SessionMessageTreeNode> {
        let active_message_ids = self
            .active_path_nodes()
            .into_iter()
            .filter_map(|node| node.message().map(|message| message.id.clone()))
            .collect::<HashSet<_>>();

        let message_nodes = self
            .nodes
            .iter()
            .filter_map(|node| {
                let message = node.message()?.clone();
                let parent_message_node_id =
                    self.nearest_message_ancestor(node.parent_node_id.as_deref());
                Some(SessionMessageTreeNode {
                    node_id: node.node_id.clone(),
                    parent_message_node_id,
                    message,
                    timestamp: node.timestamp.clone(),
                    children: Vec::new(),
                    active: active_message_ids.contains(&node.node_id),
                })
            })
            .collect::<Vec<_>>();

        build_tree(message_nodes)
    }

    fn nearest_message_ancestor(&self, node_id: Option<&str>) -> Option<String> {
        let by_id = self
            .nodes
            .iter()
            .map(|node| (node.node_id.as_str(), node))
            .collect::<HashMap<_, _>>();
        let mut current = node_id.and_then(|id| by_id.get(id).copied());
        while let Some(node) = current {
            if node.message().is_some() {
                return Some(node.node_id.clone());
            }
            current = node
                .parent_node_id
                .as_deref()
                .and_then(|parent| by_id.get(parent).copied());
        }
        None
    }
}

fn build_tree(mut nodes: Vec<SessionMessageTreeNode>) -> Vec<SessionMessageTreeNode> {
    let mut children_by_parent = HashMap::<Option<String>, Vec<SessionMessageTreeNode>>::new();
    for node in nodes.drain(..) {
        children_by_parent
            .entry(node.parent_message_node_id.clone())
            .or_default()
            .push(node);
    }
    let mut roots = build_tree_children(None, &mut children_by_parent);
    sort_tree(&mut roots);
    roots
}

fn sort_tree(nodes: &mut [SessionMessageTreeNode]) {
    nodes.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    for node in nodes {
        sort_tree(&mut node.children);
    }
}

fn build_tree_children(
    parent_id: Option<String>,
    children_by_parent: &mut HashMap<Option<String>, Vec<SessionMessageTreeNode>>,
) -> Vec<SessionMessageTreeNode> {
    let mut children = children_by_parent.remove(&parent_id).unwrap_or_default();
    for child in &mut children {
        child.children = build_tree_children(Some(child.node_id.clone()), children_by_parent);
    }
    children
}

fn build_projection_items<'a>(
    messages: &'a [Message],
    tool_calls: &'a [ToolCallRecord],
) -> Vec<ProjectionItem<'a>> {
    let mut first_message_for_call = HashMap::<String, usize>::new();
    for (idx, message) in messages.iter().enumerate() {
        for part in &message.parts {
            if let Some(call_id) = &part.tool_call_id {
                first_message_for_call.entry(call_id.clone()).or_insert(idx);
            }
        }
    }

    let mut anchored = HashMap::<usize, Vec<ProjectionItem<'a>>>::new();
    for record in tool_calls {
        let stable_key = stable_tool_call_key(record);
        let anchor = record
            .call_id
            .as_ref()
            .and_then(|call_id| first_message_for_call.get(call_id).copied())
            .unwrap_or_else(|| messages.len().saturating_sub(1));
        anchored
            .entry(anchor)
            .or_default()
            .push(ProjectionItem::ToolCall { stable_key, record });
    }

    let mut out = Vec::new();
    for (idx, message) in messages.iter().enumerate() {
        out.push(ProjectionItem::Message(message));
        if let Some(items) = anchored.remove(&idx) {
            out.extend(items);
        }
    }
    out
}

fn recognized_projection_key(node: &SessionNodeRecord) -> Option<String> {
    match &node.payload {
        SessionNodePayload::Message { message } => Some(format!("message:{}", message.id)),
        SessionNodePayload::Plugin { plugin_type, body }
            if plugin_type == INTERNAL_TOOL_CALL_PLUGIN_TYPE =>
        {
            serde_json::from_value::<ToolCallPluginBody>(body.clone())
                .ok()
                .map(|body| format!("tool_call:{}", body.stable_key))
        }
        SessionNodePayload::Plugin { .. } => None,
    }
}

fn target_item_key(item: &ProjectionItem<'_>) -> String {
    match item {
        ProjectionItem::Message(message) => format!("message:{}", message.id),
        ProjectionItem::ToolCall { stable_key, .. } => format!("tool_call:{stable_key}"),
    }
}

fn unique_plugin_node_id(stable_key: &str, existing_ids: &HashSet<String>) -> String {
    let base = format!("plugin:{}", stable_key);
    if !existing_ids.contains(&base) {
        return base;
    }
    loop {
        let candidate = format!("plugin:{}:{}", stable_key, uuid::Uuid::new_v4());
        if !existing_ids.contains(&candidate) {
            return candidate;
        }
    }
}

fn fresh_node_id(prefix: &str) -> String {
    format!("{prefix}{}", uuid::Uuid::new_v4().simple())
}

fn stable_tool_call_key(record: &ToolCallRecord) -> String {
    if let Some(call_id) = record
        .call_id
        .as_ref()
        .filter(|call_id| !call_id.is_empty())
    {
        return call_id.clone();
    }
    let raw = serde_json::to_vec(&(record.tool.clone(), &record.args, &record.result))
        .unwrap_or_else(|_| b"tool-call".to_vec());
    let digest = sha2::Sha256::digest(raw);
    format!("anon-{}", &format!("{digest:x}")[..12])
}

fn first_message_search_text(message: &Message) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| match part.kind {
            crate::PartKind::ToolCall | crate::PartKind::ToolResult => None,
            crate::PartKind::Image => Some("[Image attached]".to_string()),
            _ => (!part.content.trim().is_empty()).then(|| part.content.clone()),
        })
        .collect::<Vec<_>>()
        .join("\n\n")
        .trim()
        .to_string()
}
