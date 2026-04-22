use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::sync::{Arc, OnceLock};

use chrono::Utc;
use lash_sansio::session_model::message::{append_rendered_prompt, render_prompt};
use serde::Deserialize;
use sha2::Digest;

use crate::{
    ContextApproach, ExecutionMode, Message, MessageRole, PromptUsage, TokenUsage, ToolCallRecord,
};

pub const INTERNAL_TOOL_CALL_PLUGIN_TYPE: &str = "lash.tool_call_record";
pub const INTERNAL_RLM_GLOBALS_PATCH_PLUGIN_TYPE: &str = "lash.execution.rlm.globals_patch";

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SessionGraphData {
    #[serde(default)]
    pub nodes: Vec<SessionNodeRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leaf_node_id: Option<String>,
}

#[derive(Debug)]
pub struct SessionGraph {
    inner: Arc<SessionGraphData>,
    cache: Arc<OnceLock<SessionGraphCache>>,
}

impl Default for SessionGraph {
    fn default() -> Self {
        Self {
            inner: Arc::new(SessionGraphData::default()),
            cache: Arc::new(OnceLock::new()),
        }
    }
}

impl Clone for SessionGraph {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            cache: Arc::clone(&self.cache),
        }
    }
}

impl serde::Serialize for SessionGraph {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for SessionGraph {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let inner = SessionGraphData::deserialize(deserializer)?;
        Ok(Self {
            inner: Arc::new(inner),
            cache: Arc::new(OnceLock::new()),
        })
    }
}

impl Deref for SessionGraph {
    type Target = SessionGraphData;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SharedJsonValue(pub Arc<serde_json::Value>);

impl SharedJsonValue {
    pub fn new(value: serde_json::Value) -> Self {
        Self(Arc::new(value))
    }

    pub fn to_owned(&self) -> serde_json::Value {
        self.0.as_ref().clone()
    }
}

impl AsRef<serde_json::Value> for SharedJsonValue {
    fn as_ref(&self) -> &serde_json::Value {
        self.0.as_ref()
    }
}

impl serde::Serialize for SharedJsonValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for SharedJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        Ok(Self::new(value))
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionNodePayload {
    Message {
        message: Message,
    },
    Plugin {
        plugin_type: String,
        body: SharedJsonValue,
    },
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolCallPluginBody {
    pub stable_key: String,
    pub record: ToolCallRecord,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PersistedSessionConfig {
    pub provider_id: String,
    pub configured_model: String,
    pub context_window: u64,
    pub execution_mode: ExecutionMode,
    #[serde(default)]
    pub context_approach: ContextApproach,
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

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct RlmGlobalsPatchPluginBody {
    #[serde(default)]
    pub set: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub unset: Vec<String>,
}

impl RlmGlobalsPatchPluginBody {
    pub fn is_empty(&self) -> bool {
        self.set.is_empty() && self.unset.is_empty()
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

#[derive(Clone, Debug)]
pub(crate) struct SessionGraphAppendBuilder {
    existing_ids: HashSet<String>,
    leaf_node_id: Option<String>,
}

impl SessionGraphAppendBuilder {
    pub(crate) fn leaf_node_id(&self) -> Option<&String> {
        self.leaf_node_id.as_ref()
    }

    pub(crate) fn append_messages<I>(&mut self, messages: I) -> Vec<SessionNodeRecord>
    where
        I: IntoIterator<Item = Message>,
    {
        let mut nodes = Vec::new();
        for mut message in messages {
            if message.id.is_empty() {
                message.id = fresh_node_id("m");
            }
            let node_id = unique_message_node_id(&message.id, &self.existing_ids);
            self.existing_ids.insert(node_id.clone());
            let parent_node_id = self.leaf_node_id.clone();
            self.leaf_node_id = Some(node_id.clone());
            nodes.push(SessionNodeRecord {
                node_id,
                parent_node_id,
                timestamp: Utc::now().to_rfc3339(),
                payload: SessionNodePayload::Message { message },
            });
        }
        nodes
    }

    pub(crate) fn append_tool_call_records<I>(&mut self, records: I) -> Vec<SessionNodeRecord>
    where
        I: IntoIterator<Item = ToolCallRecord>,
    {
        let mut nodes = Vec::new();
        for record in records {
            let stable_key = stable_tool_call_key(&record);
            let node_id = unique_plugin_node_id(&stable_key, &self.existing_ids);
            self.existing_ids.insert(node_id.clone());
            let parent_node_id = self.leaf_node_id.clone();
            self.leaf_node_id = Some(node_id.clone());
            nodes.push(SessionNodeRecord {
                node_id,
                parent_node_id,
                timestamp: Utc::now().to_rfc3339(),
                payload: SessionNodePayload::Plugin {
                    plugin_type: INTERNAL_TOOL_CALL_PLUGIN_TYPE.to_string(),
                    body: SharedJsonValue::new(
                        serde_json::to_value(ToolCallPluginBody { stable_key, record })
                            .unwrap_or(serde_json::Value::Null),
                    ),
                },
            });
        }
        nodes
    }
}

#[derive(Debug, Clone)]
struct SessionGraphCache {
    by_id: HashMap<String, usize>,
    active_path_indices: Vec<usize>,
    projected_messages: Arc<Vec<Message>>,
    projected_tool_calls: Arc<Vec<ToolCallRecord>>,
    projected_rlm_globals: Arc<serde_json::Map<String, serde_json::Value>>,
    rendered_prompt: Arc<crate::RenderedPrompt>,
}

impl SessionGraphCache {
    fn build(graph: &SessionGraph) -> Self {
        let by_id = graph
            .nodes
            .iter()
            .enumerate()
            .map(|(idx, node)| (node.node_id.clone(), idx))
            .collect::<HashMap<_, _>>();
        let mut active_path_indices = Vec::new();
        let mut current = graph
            .leaf_node_id
            .as_ref()
            .and_then(|node_id| by_id.get(node_id).copied());
        while let Some(idx) = current {
            active_path_indices.push(idx);
            current = graph.nodes[idx]
                .parent_node_id
                .as_ref()
                .and_then(|node_id| by_id.get(node_id).copied());
        }
        active_path_indices.reverse();

        let mut cache = Self {
            by_id,
            active_path_indices,
            projected_messages: Arc::new(Vec::new()),
            projected_tool_calls: Arc::new(Vec::new()),
            projected_rlm_globals: Arc::new(serde_json::Map::new()),
            rendered_prompt: Arc::new(crate::RenderedPrompt::default()),
        };
        cache.rebuild_projection(graph);
        cache
    }

    fn rebuild_projection(&mut self, graph: &SessionGraph) {
        let mut projected_messages = Vec::with_capacity(self.active_path_indices.len());
        let mut projected_tool_calls = Vec::with_capacity(self.active_path_indices.len());
        let mut projected_rlm_globals = serde_json::Map::new();
        let mut seen_ids = HashSet::with_capacity(self.active_path_indices.len());
        for idx in &self.active_path_indices {
            let node = &graph.nodes[*idx];
            if let Some(message) = node.message() {
                if !message.is_transient() && seen_ids.insert(message.id.clone()) {
                    projected_messages.push(message.clone());
                }
                continue;
            }
            if let Some((INTERNAL_TOOL_CALL_PLUGIN_TYPE, body)) = node.plugin()
                && let Ok(body) = ToolCallPluginBody::deserialize(body)
            {
                projected_tool_calls.push(body.record);
                continue;
            }
            if let Some((INTERNAL_RLM_GLOBALS_PATCH_PLUGIN_TYPE, body)) = node.plugin()
                && let Ok(patch) = RlmGlobalsPatchPluginBody::deserialize(body)
            {
                apply_rlm_globals_patch_map(&mut projected_rlm_globals, patch);
            }
        }
        self.projected_messages = Arc::new(projected_messages);
        self.projected_tool_calls = Arc::new(projected_tool_calls);
        self.projected_rlm_globals = Arc::new(projected_rlm_globals);
        self.rendered_prompt = Arc::new(render_prompt(self.projected_messages.as_slice()));
    }

    fn append_node(
        &mut self,
        node_index: usize,
        node: &SessionNodeRecord,
        previous_leaf_node_id: Option<&str>,
    ) {
        self.by_id.insert(node.node_id.clone(), node_index);
        let parent_matches_leaf = node.parent_node_id.as_deref() == previous_leaf_node_id;
        if !parent_matches_leaf {
            return;
        }
        self.active_path_indices.push(node_index);
        if let Some(message) = node.message() {
            if !message.is_transient()
                && !self
                    .projected_messages
                    .iter()
                    .any(|existing| existing.id == message.id)
            {
                Arc::make_mut(&mut self.projected_messages).push(message.clone());
                append_rendered_prompt(
                    Arc::make_mut(&mut self.rendered_prompt),
                    std::slice::from_ref(message),
                );
            }
            return;
        }
        if let Some((INTERNAL_TOOL_CALL_PLUGIN_TYPE, body)) = node.plugin()
            && let Ok(body) = ToolCallPluginBody::deserialize(body)
        {
            Arc::make_mut(&mut self.projected_tool_calls).push(body.record);
            return;
        }
        if let Some((INTERNAL_RLM_GLOBALS_PATCH_PLUGIN_TYPE, body)) = node.plugin()
            && let Ok(patch) = RlmGlobalsPatchPluginBody::deserialize(body)
        {
            apply_rlm_globals_patch_map(Arc::make_mut(&mut self.projected_rlm_globals), patch);
        }
    }

    fn reserve_append_capacity(
        &mut self,
        additional_nodes: usize,
        additional_messages: usize,
        additional_tool_calls: usize,
    ) {
        self.by_id.reserve(additional_nodes);
        self.active_path_indices.reserve(additional_nodes);
        if additional_messages > 0 {
            Arc::make_mut(&mut self.projected_messages).reserve(additional_messages);
            let rendered = Arc::make_mut(&mut self.rendered_prompt);
            rendered.messages.reserve(additional_messages);
        }
        if additional_tool_calls > 0 {
            Arc::make_mut(&mut self.projected_tool_calls).reserve(additional_tool_calls);
        }
    }
}

fn apply_rlm_globals_patch_map(
    globals: &mut serde_json::Map<String, serde_json::Value>,
    patch: RlmGlobalsPatchPluginBody,
) {
    for key in patch.unset {
        globals.remove(&key);
    }
    for (key, value) in patch.set {
        globals.insert(key, value);
    }
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
            SessionNodePayload::Plugin { plugin_type, body } => {
                Some((plugin_type.as_str(), body.as_ref()))
            }
        }
    }

    pub fn plugin_body<T>(&self) -> Option<T>
    where
        T: for<'de> serde::Deserialize<'de>,
    {
        let (_, body) = self.plugin()?;
        T::deserialize(body).ok()
    }
}

impl SessionGraph {
    pub fn append_projection_delta(&mut self, messages: &[Message], tool_calls: &[ToolCallRecord]) {
        let appendable_messages = {
            let mut seen_message_ids = self
                .projected_messages()
                .iter()
                .map(|message| message.id.as_str())
                .collect::<HashSet<_>>();
            messages
                .iter()
                .filter(|message| {
                    !message.is_transient() && seen_message_ids.insert(message.id.as_str())
                })
                .cloned()
                .collect::<Vec<_>>()
        };
        let mut seen_tool_call_keys = self
            .projected_tool_calls()
            .iter()
            .map(|record| tool_call_projection_key(&stable_tool_call_key(record), record))
            .collect::<HashSet<_>>();
        let appendable_tool_calls = tool_calls
            .iter()
            .filter_map(|record| {
                let stable_key = stable_tool_call_key(record);
                let projection_key = tool_call_projection_key(&stable_key, record);
                seen_tool_call_keys
                    .insert(projection_key)
                    .then_some((stable_key, record))
            })
            .collect::<Vec<_>>();

        self.reserve_append_capacity(
            appendable_messages.len() + appendable_tool_calls.len(),
            appendable_messages.len(),
            appendable_tool_calls.len(),
        );
        self.append_message_batch(appendable_messages);

        for (stable_key, record) in appendable_tool_calls {
            self.append_plugin(
                INTERNAL_TOOL_CALL_PLUGIN_TYPE,
                serde_json::to_value(ToolCallPluginBody {
                    stable_key,
                    record: record.clone(),
                })
                .unwrap_or(serde_json::Value::Null),
            );
        }
    }

    pub(crate) fn append_projected_messages(&mut self, messages: &[Message]) {
        let appendable_messages = messages
            .iter()
            .filter(|message| !message.is_transient())
            .cloned()
            .collect::<Vec<_>>();
        self.reserve_append_capacity(appendable_messages.len(), appendable_messages.len(), 0);
        self.append_message_batch(appendable_messages);
    }

    pub fn from_nodes(nodes: Vec<SessionNodeRecord>, leaf_node_id: Option<String>) -> Self {
        Self {
            inner: Arc::new(SessionGraphData {
                nodes,
                leaf_node_id,
            }),
            cache: Arc::new(OnceLock::new()),
        }
    }

    pub(crate) fn append_builder(&self) -> SessionGraphAppendBuilder {
        SessionGraphAppendBuilder {
            existing_ids: self.nodes.iter().map(|node| node.node_id.clone()).collect(),
            leaf_node_id: self.leaf_node_id.clone(),
        }
    }

    fn invalidate_cache(&mut self) {
        self.cache = Arc::new(OnceLock::new());
    }

    fn data_mut(&mut self) -> &mut SessionGraphData {
        self.invalidate_cache();
        Arc::make_mut(&mut self.inner)
    }

    fn reserve_append_capacity(
        &mut self,
        additional_nodes: usize,
        additional_messages: usize,
        additional_tool_calls: usize,
    ) {
        if additional_nodes == 0 {
            return;
        }
        self.detach_initialized_cache_for_append();
        Arc::make_mut(&mut self.inner)
            .nodes
            .reserve(additional_nodes);
        if let Some(cache_lock) = Arc::get_mut(&mut self.cache)
            && let Some(cache) = cache_lock.get_mut()
        {
            cache.reserve_append_capacity(
                additional_nodes,
                additional_messages,
                additional_tool_calls,
            );
        }
    }

    fn detach_initialized_cache_for_append(&mut self) {
        if Arc::get_mut(&mut self.cache).is_some() {
            return;
        }
        let Some(cache) = self.cache.get().cloned() else {
            self.invalidate_cache();
            return;
        };
        let lock = OnceLock::new();
        let _ = lock.set(cache);
        self.cache = Arc::new(lock);
    }

    fn cache(&self) -> &SessionGraphCache {
        self.cache.get_or_init(|| SessionGraphCache::build(self))
    }

    fn append_message_batch(&mut self, messages: Vec<Message>) {
        if messages.is_empty() {
            return;
        }

        let messages = messages.into_iter();
        let mut existing_ids = self
            .nodes
            .iter()
            .map(|node| node.node_id.clone())
            .collect::<HashSet<_>>();
        let mut parent_node_id = self.leaf_node_id.clone();
        let mut nodes = Vec::with_capacity(messages.len());
        for mut message in messages {
            if message.id.is_empty() {
                message.id = fresh_node_id("m");
            }
            let node_id = unique_message_node_id(&message.id, &existing_ids);
            existing_ids.insert(node_id.clone());
            nodes.push(SessionNodeRecord {
                node_id: node_id.clone(),
                parent_node_id,
                timestamp: Utc::now().to_rfc3339(),
                payload: SessionNodePayload::Message { message },
            });
            parent_node_id = Some(node_id);
        }

        self.append_prebuilt_nodes(nodes);
    }

    fn append_prebuilt_nodes(&mut self, nodes: Vec<SessionNodeRecord>) {
        if nodes.is_empty() {
            return;
        }

        self.detach_initialized_cache_for_append();
        if let Some(cache_lock) = Arc::get_mut(&mut self.cache)
            && let Some(cache) = cache_lock.get_mut()
        {
            let data = Arc::make_mut(&mut self.inner);
            for node in nodes {
                let previous_leaf = data.leaf_node_id.clone();
                let node_id = node.node_id.clone();
                data.nodes.push(node);
                cache.append_node(
                    data.nodes.len() - 1,
                    data.nodes.last().expect("just appended graph node"),
                    previous_leaf.as_deref(),
                );
                data.leaf_node_id = Some(node_id);
            }
            return;
        }

        let data = self.data_mut();
        for node in nodes {
            data.leaf_node_id = Some(node.node_id.clone());
            data.nodes.push(node);
        }
    }

    pub fn append_message(&mut self, mut message: Message) -> String {
        if message.id.is_empty() {
            message.id = fresh_node_id("m");
        }
        let existing_ids = self
            .nodes
            .iter()
            .map(|node| node.node_id.clone())
            .collect::<HashSet<_>>();
        let node_id = unique_message_node_id(&message.id, &existing_ids);
        let previous_leaf = self.leaf_node_id.clone();
        let parent_node_id = previous_leaf.clone();
        let node = SessionNodeRecord {
            node_id: node_id.clone(),
            parent_node_id,
            timestamp: Utc::now().to_rfc3339(),
            payload: SessionNodePayload::Message { message },
        };
        self.detach_initialized_cache_for_append();
        if let Some(cache_lock) = Arc::get_mut(&mut self.cache)
            && let Some(cache) = cache_lock.get_mut()
        {
            let data = Arc::make_mut(&mut self.inner);
            data.nodes.push(node);
            cache.append_node(
                data.nodes.len() - 1,
                data.nodes.last().expect("just appended graph node"),
                previous_leaf.as_deref(),
            );
            data.leaf_node_id = Some(node_id.clone());
            return node_id;
        }
        let data = self.data_mut();
        data.nodes.push(node);
        data.leaf_node_id = Some(node_id.clone());
        node_id
    }

    pub fn append_plugin(
        &mut self,
        plugin_type: impl Into<String>,
        body: serde_json::Value,
    ) -> String {
        let node_id = fresh_node_id("x");
        let previous_leaf = self.leaf_node_id.clone();
        let parent_node_id = previous_leaf.clone();
        let node = SessionNodeRecord {
            node_id: node_id.clone(),
            parent_node_id,
            timestamp: Utc::now().to_rfc3339(),
            payload: SessionNodePayload::Plugin {
                plugin_type: plugin_type.into(),
                body: SharedJsonValue::new(body),
            },
        };
        self.detach_initialized_cache_for_append();
        if let Some(cache_lock) = Arc::get_mut(&mut self.cache)
            && let Some(cache) = cache_lock.get_mut()
        {
            let data = Arc::make_mut(&mut self.inner);
            data.nodes.push(node);
            cache.append_node(
                data.nodes.len() - 1,
                data.nodes.last().expect("just appended graph node"),
                previous_leaf.as_deref(),
            );
            data.leaf_node_id = Some(node_id.clone());
            return node_id;
        }
        let data = self.data_mut();
        data.nodes.push(node);
        data.leaf_node_id = Some(node_id.clone());
        node_id
    }

    pub fn active_path_nodes(&self) -> Vec<&SessionNodeRecord> {
        self.cache()
            .active_path_indices
            .iter()
            .map(|idx| &self.nodes[*idx])
            .collect()
    }

    pub fn project_messages(&self) -> Vec<Message> {
        self.cache().projected_messages.as_ref().clone()
    }

    pub fn projected_messages(&self) -> &[Message] {
        self.cache().projected_messages.as_slice()
    }

    pub fn shared_projected_messages(&self) -> Arc<Vec<Message>> {
        Arc::clone(&self.cache().projected_messages)
    }

    pub fn shared_projected_rendered_prompt(&self) -> Arc<crate::RenderedPrompt> {
        Arc::clone(&self.cache().rendered_prompt)
    }

    pub fn project_tool_calls(&self) -> Vec<ToolCallRecord> {
        self.cache().projected_tool_calls.as_ref().clone()
    }

    pub fn projected_tool_calls(&self) -> &[ToolCallRecord] {
        self.cache().projected_tool_calls.as_slice()
    }

    pub fn shared_projected_tool_calls(&self) -> Arc<Vec<ToolCallRecord>> {
        Arc::clone(&self.cache().projected_tool_calls)
    }

    pub fn shared_projected_rlm_globals(&self) -> Arc<serde_json::Map<String, serde_json::Value>> {
        Arc::clone(&self.cache().projected_rlm_globals)
    }

    pub fn replace_tool_call_projection(&mut self, tool_calls: &[ToolCallRecord]) {
        let messages = Arc::clone(&self.cache().projected_messages);
        self.merge_active_projection(messages.as_slice(), tool_calls);
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

    pub fn projected_rlm_globals(&self) -> serde_json::Map<String, serde_json::Value> {
        self.cache().projected_rlm_globals.as_ref().clone()
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
        self.data_mut().leaf_node_id = node_id;
    }

    pub fn set_leaf_node_id(&mut self, node_id: Option<String>) {
        self.data_mut().leaf_node_id = node_id;
    }

    pub fn push_node_record(&mut self, node: SessionNodeRecord) {
        self.data_mut().nodes.push(node);
    }

    pub fn extend_node_records<I>(&mut self, nodes: I)
    where
        I: IntoIterator<Item = SessionNodeRecord>,
    {
        self.data_mut().nodes.extend(nodes);
    }

    pub fn active_path_contains(&self, node_id: &str) -> bool {
        self.active_path_nodes()
            .into_iter()
            .any(|node| node.node_id == node_id)
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
            self.data_mut().leaf_node_id = fallback;
            return true;
        }
        false
    }

    pub fn fork_current_path(&self) -> SessionGraph {
        let path = self.active_path_nodes();
        SessionGraph::from_nodes(
            path.into_iter().cloned().collect(),
            self.leaf_node_id.clone(),
        )
    }

    pub fn find_node(&self, node_id: &str) -> Option<&SessionNodeRecord> {
        self.cache()
            .by_id
            .get(node_id)
            .and_then(|idx| self.nodes.get(*idx))
    }

    pub fn node_index(&self, node_id: &str) -> Option<usize> {
        self.cache().by_id.get(node_id).copied()
    }

    pub fn merge_active_projection(&mut self, messages: &[Message], tool_calls: &[ToolCallRecord]) {
        let current_nodes = self.active_path_nodes();
        let target = build_projection_items(messages, tool_calls);

        let mut preserved_ids = Vec::new();
        let mut seen_projection_keys = HashSet::new();
        let mut target_idx = 0usize;
        for node in current_nodes {
            if node
                .message()
                .map(|message| message.is_transient())
                .unwrap_or(false)
            {
                continue;
            }
            if let Some(key) = recognized_projection_key(node) {
                if !seen_projection_keys.insert(key.clone()) {
                    continue;
                }
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

        self.data_mut().leaf_node_id = preserved_ids.last().cloned();
        let mut existing_ids = self
            .nodes
            .iter()
            .map(|node| node.node_id.clone())
            .collect::<HashSet<_>>();

        for item in target.into_iter().skip(target_idx) {
            let parent_node_id = self.leaf_node_id.clone();
            let node = match item {
                ProjectionItem::Message(message) => {
                    let node_id = unique_message_node_id(&message.id, &existing_ids);
                    SessionNodeRecord {
                        node_id,
                        parent_node_id,
                        timestamp: Utc::now().to_rfc3339(),
                        payload: SessionNodePayload::Message {
                            message: message.clone(),
                        },
                    }
                }
                ProjectionItem::ToolCall { stable_key, record } => {
                    let node_id = unique_plugin_node_id(&stable_key, &existing_ids);
                    SessionNodeRecord {
                        node_id,
                        parent_node_id,
                        timestamp: Utc::now().to_rfc3339(),
                        payload: SessionNodePayload::Plugin {
                            plugin_type: INTERNAL_TOOL_CALL_PLUGIN_TYPE.to_string(),
                            body: SharedJsonValue::new(
                                serde_json::to_value(ToolCallPluginBody {
                                    stable_key,
                                    record: record.clone(),
                                })
                                .unwrap_or(serde_json::Value::Null),
                            ),
                        },
                    }
                }
            };
            existing_ids.insert(node.node_id.clone());
            let data = self.data_mut();
            data.leaf_node_id = Some(node.node_id.clone());
            data.nodes.push(node);
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
    let projected_messages = messages
        .iter()
        .filter(|message| !message.is_transient())
        .collect::<Vec<_>>();
    for (idx, message) in projected_messages.iter().enumerate() {
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
            .unwrap_or_else(|| projected_messages.len().saturating_sub(1));
        anchored
            .entry(anchor)
            .or_default()
            .push(ProjectionItem::ToolCall { stable_key, record });
    }

    let mut out = Vec::new();
    for (idx, message) in projected_messages.iter().enumerate() {
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
            serde_json::from_value::<ToolCallPluginBody>(body.to_owned())
                .ok()
                .map(|body| tool_call_projection_key(&body.stable_key, &body.record))
        }
        SessionNodePayload::Plugin { .. } => None,
    }
}

fn target_item_key(item: &ProjectionItem<'_>) -> String {
    match item {
        ProjectionItem::Message(message) => format!("message:{}", message.id),
        ProjectionItem::ToolCall { stable_key, record } => {
            tool_call_projection_key(stable_key, record)
        }
    }
}

fn tool_call_projection_key(stable_key: &str, record: &ToolCallRecord) -> String {
    let fingerprint = serde_json::to_string(record).unwrap_or_default();
    format!("tool_call:{stable_key}:{fingerprint}")
}

pub(crate) fn tool_call_record_projection_key(record: &ToolCallRecord) -> String {
    let stable_key = stable_tool_call_key(record);
    tool_call_projection_key(&stable_key, record)
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

fn unique_message_node_id(message_id: &str, existing_ids: &HashSet<String>) -> String {
    if !existing_ids.contains(message_id) {
        return message_id.to_string();
    }
    let base = format!("message:{message_id}");
    if !existing_ids.contains(&base) {
        return base;
    }
    for suffix in 2.. {
        let candidate = format!("{base}:{suffix}");
        if !existing_ids.contains(&candidate) {
            return candidate;
        }
    }
    unreachable!("message node id space exhausted")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MessageRole, Part, PartKind, PruneState};

    fn text_message(id: &str, role: MessageRole, content: &str) -> Message {
        Message {
            id: id.to_string(),
            role,
            parts: vec![Part {
                id: format!("{id}.p0"),
                kind: PartKind::Text,
                content: content.to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
            }],
            user_input: None,
            origin: None,
        }
    }

    fn transient_plugin_message(id: &str, role: MessageRole, content: &str) -> Message {
        Message {
            origin: Some(crate::MessageOrigin::Plugin {
                plugin_id: "test-plugin".to_string(),
                transient: true,
            }),
            ..text_message(id, role, content)
        }
    }

    #[test]
    fn transient_messages_are_not_projected_or_persisted() {
        let mut graph = SessionGraph::default();
        let user = text_message("m1", MessageRole::User, "hello");
        let transient = transient_plugin_message("mem", MessageRole::System, "memory");
        let assistant = text_message("m2", MessageRole::Assistant, "world");

        graph.merge_active_projection(&[user.clone(), transient, assistant.clone()], &[]);

        let projected = graph.project_messages();
        assert_eq!(
            projected.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            vec!["m1", "m2"]
        );
        assert!(!graph.nodes.iter().any(|node| node.node_id == "mem"));
    }

    #[test]
    fn merge_active_projection_drops_duplicate_active_message_ids() {
        let mut graph = SessionGraph::default();
        let user = text_message("m1", MessageRole::User, "hello");
        let assistant = text_message("m2", MessageRole::Assistant, "world");
        graph.merge_active_projection(&[user.clone(), assistant.clone()], &[]);

        graph.push_node_record(SessionNodeRecord {
            node_id: "m2".to_string(),
            parent_node_id: Some("m1".to_string()),
            timestamp: Utc::now().to_rfc3339(),
            payload: SessionNodePayload::Message {
                message: assistant.clone(),
            },
        });
        graph.set_leaf_node_id(Some("m2".to_string()));

        graph.merge_active_projection(&[user, assistant], &[]);

        let projected = graph.project_messages();
        assert_eq!(
            projected.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            vec!["m1", "m2"]
        );
        let m2_count = graph
            .nodes
            .iter()
            .filter(|node| node.node_id == "m2")
            .count();
        assert_eq!(m2_count, 2);
    }
}
