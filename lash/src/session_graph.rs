use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::sync::{Arc, OnceLock};

use chrono::Utc;
use sha2::Digest;

use crate::session_model::{ConversationRecord, SessionEventRecord, ToolEvent};
use crate::{
    BaseRenderCache, ExecutionMode, Message, MessageRole, PromptUsage, StandardContextApproach,
    TokenUsage, ToolCallRecord,
};

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
    Event {
        event: SessionEventRecord,
    },
    Plugin {
        plugin_type: String,
        body: SharedJsonValue,
    },
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PersistedSessionConfig {
    pub provider_id: String,
    pub configured_model: String,
    pub context_window: u64,
    pub execution_mode: ExecutionMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standard_context_approach: Option<StandardContextApproach>,
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
    #[serde(default)]
    pub mode_turn_options: crate::ModeTurnOptions,
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
enum ActiveReadItem<'a> {
    Message(&'a Message),
    ToolCall {
        stable_key: String,
        record: &'a ToolCallRecord,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct SessionReadModel {
    pub(crate) active_events: Arc<Vec<SessionEventRecord>>,
    pub(crate) messages: Arc<Vec<Message>>,
    pub(crate) tool_calls: Arc<Vec<ToolCallRecord>>,
    pub(crate) rlm_globals: Arc<serde_json::Map<String, serde_json::Value>>,
    pub(crate) prompt_render_cache: Arc<BaseRenderCache>,
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
                payload: SessionNodePayload::Event {
                    event: SessionEventRecord::Conversation(ConversationRecord::from_message(
                        message,
                    )),
                },
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
                payload: SessionNodePayload::Event {
                    event: SessionEventRecord::Tool(ToolEvent::Invocation { stable_key, record }),
                },
            });
        }
        nodes
    }

    pub(crate) fn append_event_record(
        &mut self,
        event: SessionEventRecord,
    ) -> Vec<SessionNodeRecord> {
        let node_id = unique_plugin_node_id(&fresh_node_id("e"), &self.existing_ids);
        self.existing_ids.insert(node_id.clone());
        let parent_node_id = self.leaf_node_id.clone();
        self.leaf_node_id = Some(node_id.clone());
        vec![SessionNodeRecord {
            node_id,
            parent_node_id,
            timestamp: Utc::now().to_rfc3339(),
            payload: SessionNodePayload::Event { event },
        }]
    }
}

#[derive(Debug, Clone)]
struct SessionGraphCache {
    by_id: HashMap<String, usize>,
    active_path_indices: Vec<usize>,
    active_events: Arc<Vec<SessionEventRecord>>,
    active_messages: Arc<Vec<Message>>,
    /// Index from `Message::id` to its position in `active_messages`,
    /// kept in sync with the vec so dedup on append is O(1) instead of an
    /// O(n) linear scan (which made long sessions quadratic in message
    /// count).
    active_message_ids: HashMap<String, usize>,
    active_tool_calls: Arc<Vec<ToolCallRecord>>,
    /// RLM globals state. Maintained incrementally on append so the
    /// per-iteration `bound_variables` prompt contribution doesn't replay
    /// every patch event in the active path.
    rlm_globals: Arc<serde_json::Map<String, serde_json::Value>>,
    /// Memoized render of `active_messages`. Shared with every
    /// `MessageSequence` built off this read model so the chat projector's
    /// per-iteration `render_prompt` walk only happens once per turn.
    /// Replaced (not invalidated in-place) whenever `active_messages`
    /// changes — the `Arc` identity tracks the cache's validity.
    prompt_render_cache: Arc<BaseRenderCache>,
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
            active_events: Arc::new(Vec::new()),
            active_messages: Arc::new(Vec::new()),
            active_message_ids: HashMap::new(),
            active_tool_calls: Arc::new(Vec::new()),
            rlm_globals: Arc::new(serde_json::Map::new()),
            prompt_render_cache: Arc::new(BaseRenderCache::new()),
        };
        cache.rebuild_read_model(graph);
        cache
    }

    fn rebuild_read_model(&mut self, graph: &SessionGraph) {
        let mut active_messages = Vec::with_capacity(self.active_path_indices.len());
        let mut active_message_ids: HashMap<String, usize> =
            HashMap::with_capacity(self.active_path_indices.len());
        let mut active_tool_calls = Vec::with_capacity(self.active_path_indices.len());
        let mut active_events = Vec::with_capacity(self.active_path_indices.len());
        for idx in &self.active_path_indices {
            let node = &graph.nodes[*idx];
            if let Some(event) = node.event() {
                active_events.push(event.clone());
            }
            if let Some(message) = node.message() {
                if !message.is_transient() && !active_message_ids.contains_key(&message.id) {
                    active_message_ids.insert(message.id.clone(), active_messages.len());
                    active_messages.push(message);
                }
                continue;
            }
            if let Some(event) = node.event()
                && let SessionEventRecord::Tool(ToolEvent::Invocation { record, .. }) = event
            {
                active_tool_calls.push(record.clone());
                continue;
            }
        }
        let rlm_globals =
            crate::chronological::project_rlm_globals_from_events(active_events.iter());
        self.active_messages = Arc::new(active_messages);
        self.active_message_ids = active_message_ids;
        self.active_events = Arc::new(active_events);
        self.active_tool_calls = Arc::new(active_tool_calls);
        self.rlm_globals = Arc::new(rlm_globals);
        self.prompt_render_cache = Arc::new(BaseRenderCache::new());
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
        if let Some(event) = node.event() {
            Arc::make_mut(&mut self.active_events).push(event.clone());
            self.rlm_globals = Arc::new(crate::chronological::project_rlm_globals_from_events(
                self.active_events.iter(),
            ));
        }
        if let Some(message) = node.message() {
            if !message.is_transient() && !self.active_message_ids.contains_key(&message.id) {
                let messages = Arc::make_mut(&mut self.active_messages);
                self.active_message_ids
                    .insert(message.id.clone(), messages.len());
                messages.push(message);
                self.prompt_render_cache = Arc::new(BaseRenderCache::new());
            }
            return;
        }
        if let Some(event) = node.event()
            && let SessionEventRecord::Tool(ToolEvent::Invocation { record, .. }) = event
        {
            Arc::make_mut(&mut self.active_tool_calls).push(record.clone());
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
            Arc::make_mut(&mut self.active_messages).reserve(additional_messages);
        }
        if additional_tool_calls > 0 {
            Arc::make_mut(&mut self.active_tool_calls).reserve(additional_tool_calls);
        }
    }
}

impl SessionNodeRecord {
    pub fn event(&self) -> Option<&SessionEventRecord> {
        match &self.payload {
            SessionNodePayload::Event { event } => Some(event),
            SessionNodePayload::Plugin { .. } => None,
        }
    }

    pub fn message(&self) -> Option<Message> {
        match self.event()? {
            SessionEventRecord::Conversation(record) => Some(record.to_message()),
            _ => None,
        }
    }

    pub fn plugin(&self) -> Option<(&str, &serde_json::Value)> {
        match &self.payload {
            SessionNodePayload::Event { .. } => None,
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
    pub fn append_active_read_delta(
        &mut self,
        messages: &[Message],
        tool_calls: &[ToolCallRecord],
    ) {
        let appendable_messages = {
            let read_model = self.read_model();
            let mut seen_message_ids = read_model
                .messages
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
        let read_model = self.read_model();
        let mut seen_tool_call_keys = read_model
            .tool_calls
            .iter()
            .map(|record| tool_call_active_read_key(&stable_tool_call_key(record), record))
            .collect::<HashSet<_>>();
        let appendable_tool_calls = tool_calls
            .iter()
            .filter_map(|record| {
                let stable_key = stable_tool_call_key(record);
                let active_read_key = tool_call_active_read_key(&stable_key, record);
                seen_tool_call_keys
                    .insert(active_read_key)
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
            self.append_event(SessionEventRecord::Tool(ToolEvent::Invocation {
                stable_key,
                record: record.clone(),
            }));
        }
    }

    pub(crate) fn append_active_conversation_messages(&mut self, messages: &[Message]) {
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
                payload: SessionNodePayload::Event {
                    event: SessionEventRecord::Conversation(ConversationRecord::from_message(
                        message,
                    )),
                },
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
            payload: SessionNodePayload::Event {
                event: SessionEventRecord::Conversation(ConversationRecord::from_message(message)),
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

    pub(crate) fn read_model(&self) -> SessionReadModel {
        let cache = self.cache();
        SessionReadModel {
            active_events: Arc::clone(&cache.active_events),
            messages: Arc::clone(&cache.active_messages),
            tool_calls: Arc::clone(&cache.active_tool_calls),
            rlm_globals: Arc::clone(&cache.rlm_globals),
            prompt_render_cache: Arc::clone(&cache.prompt_render_cache),
        }
    }

    pub fn replace_active_tool_calls(&mut self, tool_calls: &[ToolCallRecord]) {
        let messages = Arc::clone(&self.cache().active_messages);
        self.replace_active_read_state(messages.as_slice(), tool_calls);
    }

    pub fn append_event(&mut self, event: SessionEventRecord) -> String {
        let node_id = fresh_node_id("e");
        let previous_leaf = self.leaf_node_id.clone();
        let node = SessionNodeRecord {
            node_id: node_id.clone(),
            parent_node_id: previous_leaf.clone(),
            timestamp: Utc::now().to_rfc3339(),
            payload: SessionNodePayload::Event { event },
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

    pub fn append_events<I>(&mut self, events: I) -> Vec<String>
    where
        I: IntoIterator<Item = SessionEventRecord>,
    {
        events
            .into_iter()
            .map(|event| self.append_event(event))
            .collect()
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
            .map(|message| first_message_search_text(&message))
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

    /// Append nodes that extend the current active path, advancing the
    /// leaf to the last node and updating the cache incrementally
    /// instead of invalidating it. Use this when the appended nodes are
    /// genuinely new descendants of the current leaf — e.g. the
    /// turn-driver merging `TurnGraphOverlay` deltas into the base graph.
    /// Use `extend_node_records` + `set_leaf_node_id` for store-side
    /// replay paths that don't follow the active-path append shape.
    pub fn extend_active_path(&mut self, nodes: Vec<SessionNodeRecord>) {
        self.append_prebuilt_nodes(nodes);
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

    pub fn replace_active_read_state(
        &mut self,
        messages: &[Message],
        tool_calls: &[ToolCallRecord],
    ) {
        let current_nodes = self.active_path_nodes();
        let target = build_active_read_items(messages, tool_calls);

        let mut preserved_ids = Vec::new();
        let mut seen_active_read_keys = HashSet::new();
        let mut target_idx = 0usize;
        for node in current_nodes {
            if node
                .message()
                .map(|message| message.is_transient())
                .unwrap_or(false)
            {
                continue;
            }
            if let Some(key) = recognized_active_read_key(node) {
                if !seen_active_read_keys.insert(key.clone()) {
                    continue;
                }
                let Some(target_item) = target.get(target_idx) else {
                    break;
                };
                if key != active_read_item_key(target_item) {
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
                ActiveReadItem::Message(message) => {
                    let node_id = unique_message_node_id(&message.id, &existing_ids);
                    SessionNodeRecord {
                        node_id,
                        parent_node_id,
                        timestamp: Utc::now().to_rfc3339(),
                        payload: SessionNodePayload::Event {
                            event: SessionEventRecord::Conversation(
                                ConversationRecord::from_message(message.clone()),
                            ),
                        },
                    }
                }
                ActiveReadItem::ToolCall { stable_key, record } => {
                    let node_id = unique_plugin_node_id(&stable_key, &existing_ids);
                    SessionNodeRecord {
                        node_id,
                        parent_node_id,
                        timestamp: Utc::now().to_rfc3339(),
                        payload: SessionNodePayload::Event {
                            event: SessionEventRecord::Tool(ToolEvent::Invocation {
                                stable_key,
                                record: record.clone(),
                            }),
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

    pub fn from_active_read_state(messages: &[Message], tool_calls: &[ToolCallRecord]) -> Self {
        let mut graph = Self::default();
        graph.replace_active_read_state(messages, tool_calls);
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

fn build_active_read_items<'a>(
    messages: &'a [Message],
    tool_calls: &'a [ToolCallRecord],
) -> Vec<ActiveReadItem<'a>> {
    let mut first_message_for_call = HashMap::<String, usize>::new();
    let active_messages = messages
        .iter()
        .filter(|message| !message.is_transient())
        .collect::<Vec<_>>();
    for (idx, message) in active_messages.iter().enumerate() {
        for part in message.parts.iter() {
            if let Some(call_id) = &part.tool_call_id {
                first_message_for_call.entry(call_id.clone()).or_insert(idx);
            }
        }
    }

    let mut anchored = HashMap::<usize, Vec<ActiveReadItem<'a>>>::new();
    for record in tool_calls {
        let stable_key = stable_tool_call_key(record);
        let anchor = record
            .call_id
            .as_ref()
            .and_then(|call_id| first_message_for_call.get(call_id).copied())
            .unwrap_or_else(|| active_messages.len().saturating_sub(1));
        anchored
            .entry(anchor)
            .or_default()
            .push(ActiveReadItem::ToolCall { stable_key, record });
    }

    let mut out = Vec::new();
    for (idx, message) in active_messages.iter().enumerate() {
        out.push(ActiveReadItem::Message(message));
        if let Some(items) = anchored.remove(&idx) {
            out.extend(items);
        }
    }
    out
}

fn recognized_active_read_key(node: &SessionNodeRecord) -> Option<String> {
    match &node.payload {
        SessionNodePayload::Event { event } => match event {
            SessionEventRecord::Conversation(record) => Some(format!("message:{}", record.id)),
            SessionEventRecord::Tool(ToolEvent::Invocation { stable_key, record }) => {
                Some(tool_call_active_read_key(stable_key, record))
            }
            _ => None,
        },
        SessionNodePayload::Plugin { .. } => None,
    }
}

fn active_read_item_key(item: &ActiveReadItem<'_>) -> String {
    match item {
        ActiveReadItem::Message(message) => format!("message:{}", message.id),
        ActiveReadItem::ToolCall { stable_key, record } => {
            tool_call_active_read_key(stable_key, record)
        }
    }
}

fn tool_call_active_read_key(stable_key: &str, record: &ToolCallRecord) -> String {
    let fingerprint = serde_json::to_string(record).unwrap_or_default();
    format!("tool_call:{stable_key}:{fingerprint}")
}

pub(crate) fn tool_call_record_active_read_key(record: &ToolCallRecord) -> String {
    let stable_key = stable_tool_call_key(record);
    tool_call_active_read_key(&stable_key, record)
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
    use crate::session_model::message::PartAttachment;
    use crate::{
        AttachmentId, AttachmentRef, ImageMediaType, MediaType, MessageRole, ModeEvent, Part,
        PartKind, PruneState, SessionNodePayload, ToolCallRecord,
    };
    use lash_rlm_types::{
        RlmDiagnosticEvent, RlmGlobalsPatchPluginBody, RlmModeEvent, RlmTrajectoryEntry,
    };

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
                response_meta: None,
            }]
            .into(),
            user_input: None,
            origin: None,
        }
    }

    fn image_part(id: &str) -> Part {
        Part {
            id: id.to_string(),
            kind: PartKind::Image,
            content: "plot".to_string(),
            attachment: Some(PartAttachment {
                reference: AttachmentRef {
                    id: AttachmentId::new("att-1"),
                    media_type: MediaType::Image(ImageMediaType::Png),
                    byte_len: 128,
                    width: Some(8),
                    height: Some(8),
                    label: Some("plot".to_string()),
                },
            }),
            tool_call_id: None,
            tool_name: None,
            tool_item_id: None,
            tool_signature: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
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
    fn transient_messages_are_not_read_or_persisted() {
        let mut graph = SessionGraph::default();
        let user = text_message("m1", MessageRole::User, "hello");
        let transient = transient_plugin_message("mem", MessageRole::System, "memory");
        let assistant = text_message("m2", MessageRole::Assistant, "world");

        graph.replace_active_read_state(&[user.clone(), transient, assistant.clone()], &[]);

        let read_model = graph.read_model();
        let active_messages = read_model.messages.as_slice();
        assert_eq!(
            active_messages
                .iter()
                .map(|m| m.id.as_str())
                .collect::<Vec<_>>(),
            vec!["m1", "m2"]
        );
        assert!(!graph.nodes.iter().any(|node| node.node_id == "mem"));
    }

    #[test]
    fn replace_active_read_state_drops_duplicate_active_message_ids() {
        let mut graph = SessionGraph::default();
        let user = text_message("m1", MessageRole::User, "hello");
        let assistant = text_message("m2", MessageRole::Assistant, "world");
        graph.replace_active_read_state(&[user.clone(), assistant.clone()], &[]);

        graph.push_node_record(SessionNodeRecord {
            node_id: "m2".to_string(),
            parent_node_id: Some("m1".to_string()),
            timestamp: Utc::now().to_rfc3339(),
            payload: SessionNodePayload::Event {
                event: SessionEventRecord::Conversation(ConversationRecord::from_message(
                    assistant.clone(),
                )),
            },
        });
        graph.set_leaf_node_id(Some("m2".to_string()));

        graph.replace_active_read_state(&[user, assistant], &[]);

        let read_model = graph.read_model();
        let active_messages = read_model.messages.as_slice();
        assert_eq!(
            active_messages
                .iter()
                .map(|m| m.id.as_str())
                .collect::<Vec<_>>(),
            vec!["m1", "m2"]
        );
        let m2_count = graph
            .nodes
            .iter()
            .filter(|node| node.node_id == "m2")
            .count();
        assert_eq!(m2_count, 2);
    }

    #[test]
    fn read_model_preserves_active_state_and_prompt_cache_identity() {
        let mut graph = SessionGraph::default();
        let user = text_message("m1", MessageRole::User, "hello");
        let assistant = text_message("m2", MessageRole::Assistant, "world");
        let tool_call = ToolCallRecord {
            call_id: Some("call_1".to_string()),
            tool: "lookup".to_string(),
            args: serde_json::json!({"q": "hello"}),
            result: serde_json::json!({"answer": "world"}),
            success: true,
            duration_ms: 3,
        };
        graph.replace_active_read_state(
            &[user.clone(), assistant.clone()],
            std::slice::from_ref(&tool_call),
        );
        graph.append_event(SessionEventRecord::Mode(ModeEvent::rlm(
            RlmModeEvent::RlmGlobalsPatch(RlmGlobalsPatchPluginBody {
                set: serde_json::Map::from_iter([(
                    "topic".to_string(),
                    serde_json::json!("read-model"),
                )]),
                unset: Vec::new(),
            }),
        )));

        let first = graph.read_model();
        let second = graph.read_model();
        assert!(Arc::ptr_eq(
            &first.prompt_render_cache,
            &second.prompt_render_cache
        ));
        assert_eq!(first.active_events.len(), 4);
        assert_eq!(
            first
                .messages
                .iter()
                .map(|message| message.id.as_str())
                .collect::<Vec<_>>(),
            vec![user.id.as_str(), assistant.id.as_str()]
        );
        assert_eq!(first.tool_calls.as_slice(), &[tool_call]);
        assert_eq!(
            first.rlm_globals.get("topic"),
            Some(&serde_json::json!("read-model"))
        );

        graph.append_message(text_message("m3", MessageRole::User, "again"));
        let updated = graph.read_model();
        assert!(!Arc::ptr_eq(
            &first.prompt_render_cache,
            &updated.prompt_render_cache
        ));
    }

    #[test]
    fn chronological_projection_covers_active_path_without_branch_or_transients() {
        let mut graph = SessionGraph::default();
        let mut user = text_message("u1", MessageRole::User, "first");
        Arc::make_mut(&mut user.parts).push(image_part("u1.p1"));
        let system = text_message("s1", MessageRole::System, "system");
        let assistant = text_message("a1", MessageRole::Assistant, "done");
        let transient = transient_plugin_message("transient", MessageRole::System, "hidden");
        let tool_call = ToolCallRecord {
            call_id: Some("call_1".to_string()),
            tool: "lookup".to_string(),
            args: serde_json::json!({"q": "first"}),
            result: serde_json::json!({"answer": "done"}),
            success: true,
            duration_ms: 9,
        };

        graph.append_message(user);
        graph.append_message(system);
        graph.append_message(transient);
        graph.append_event(SessionEventRecord::Mode(ModeEvent::rlm(
            RlmModeEvent::RlmTrajectoryEntry(RlmTrajectoryEntry {
                id: "rlm_step_0".to_string(),
                iteration: 0,
                reasoning: "think".to_string(),
                code: "x = 1".to_string(),
                output: vec!["observed".to_string()],
                tool_calls: Vec::new(),
                images: Vec::new(),
                error: None,
                final_output: None,
            }),
        )));
        graph.append_event(SessionEventRecord::Tool(ToolEvent::Invocation {
            stable_key: "call_1".to_string(),
            record: tool_call.clone(),
        }));
        graph.append_event(SessionEventRecord::Mode(ModeEvent::rlm(
            RlmModeEvent::RlmGlobalsPatch(RlmGlobalsPatchPluginBody {
                set: serde_json::Map::from_iter([(
                    "topic".to_string(),
                    serde_json::json!("chronology"),
                )]),
                unset: Vec::new(),
            }),
        )));
        graph.append_event(SessionEventRecord::Mode(ModeEvent::rlm(
            RlmModeEvent::RlmDiagnostic(RlmDiagnosticEvent {
                phase: "ignored".to_string(),
                payload: serde_json::json!({"debug": true}),
            }),
        )));
        graph.append_message(assistant);
        graph.push_node_record(SessionNodeRecord {
            node_id: "inactive".to_string(),
            parent_node_id: Some("u1".to_string()),
            timestamp: Utc::now().to_rfc3339(),
            payload: SessionNodePayload::Event {
                event: SessionEventRecord::Conversation(ConversationRecord::from_message(
                    text_message("branch", MessageRole::Assistant, "inactive"),
                )),
            },
        });

        let read_model = graph.read_model();
        let projection = crate::ChronologicalProjection::from_read_model(&read_model);
        let labels = projection
            .entries()
            .iter()
            .map(|entry| match &entry.payload {
                crate::ChronologicalPayload::Message(message) => match message.role {
                    MessageRole::User => format!("message:user:{}", message.id),
                    MessageRole::System => format!("message:system:{}", message.id),
                    MessageRole::Assistant => format!("message:assistant:{}", message.id),
                },
                crate::ChronologicalPayload::RlmStep(step) => format!("rlm:{}", step.iteration),
                crate::ChronologicalPayload::ToolCall(record) => format!("tool:{}", record.tool),
            })
            .collect::<Vec<_>>();

        assert_eq!(
            labels,
            vec![
                "message:user:u1",
                "message:system:s1",
                "rlm:0",
                "tool:lookup",
                "message:assistant:a1"
            ]
        );
        assert_eq!(
            projection
                .entries()
                .iter()
                .map(|entry| entry.index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
        assert_eq!(projection.tool_call_by_call_id("call_1"), Some(&tool_call));
        let history = projection.rlm_history();
        assert_eq!(history.len(), projection.entries().len());
        match &history[0] {
            lash_rlm_types::RlmHistoryItem::Message { attachments, .. } => {
                assert_eq!(attachments.len(), 1);
                assert_eq!(attachments[0].reference, "att-1");
            }
            other => panic!("expected first history item to be a message, got {other:?}"),
        }
        assert_eq!(
            read_model.rlm_globals.get("topic"),
            Some(&serde_json::json!("chronology"))
        );
        assert!(read_model.rlm_globals.get("history").is_none());
        assert!(!labels.iter().any(|label| label.contains("inactive")));
        assert!(!labels.iter().any(|label| label.contains("transient")));
    }

    #[test]
    fn chronological_projection_normalizes_duplicate_tool_calls_inside_rlm_steps() {
        let mut graph = SessionGraph::default();
        let shell_call = ToolCallRecord {
            call_id: Some("call_shell".to_string()),
            tool: "exec_command".to_string(),
            args: serde_json::json!({"cmd": "date"}),
            result: serde_json::json!({"output": "now\n", "exit_code": 0}),
            success: true,
            duration_ms: 7,
        };
        let lookup_call = ToolCallRecord {
            call_id: Some("call_lookup".to_string()),
            tool: "lookup".to_string(),
            args: serde_json::json!({"q": "now"}),
            result: serde_json::json!({"answer": "now"}),
            success: true,
            duration_ms: 4,
        };

        graph.append_event(SessionEventRecord::Tool(ToolEvent::Invocation {
            stable_key: "call_shell".to_string(),
            record: shell_call.clone(),
        }));
        graph.append_event(SessionEventRecord::Mode(ModeEvent::rlm(
            RlmModeEvent::RlmTrajectoryEntry(RlmTrajectoryEntry {
                id: "rlm_step_1".to_string(),
                iteration: 1,
                reasoning: "inspect".to_string(),
                code: "print now".to_string(),
                output: vec!["now".to_string()],
                tool_calls: vec![shell_call.clone(), lookup_call.clone(), lookup_call.clone()],
                images: Vec::new(),
                error: None,
                final_output: None,
            }),
        )));
        graph.append_event(SessionEventRecord::Tool(ToolEvent::Invocation {
            stable_key: "call_lookup".to_string(),
            record: lookup_call,
        }));

        let projection = crate::ChronologicalProjection::from_read_model(&graph.read_model());
        let tool_counts = projection
            .entries()
            .iter()
            .map(|entry| match &entry.payload {
                crate::ChronologicalPayload::ToolCall(_) => 1,
                crate::ChronologicalPayload::RlmStep(entry) => entry.tool_calls.len(),
                crate::ChronologicalPayload::Message(_) => 0,
            })
            .sum::<usize>();

        assert_eq!(tool_counts, 2);
        assert_eq!(
            projection
                .entries()
                .iter()
                .map(|entry| match &entry.payload {
                    crate::ChronologicalPayload::ToolCall(record) => record.tool.as_str(),
                    crate::ChronologicalPayload::RlmStep(entry) =>
                        entry.tool_calls[0].tool.as_str(),
                    crate::ChronologicalPayload::Message(_) => "message",
                })
                .collect::<Vec<_>>(),
            vec!["exec_command", "lookup"]
        );
        assert_eq!(projection.rlm_history().len(), 2);
    }
}
