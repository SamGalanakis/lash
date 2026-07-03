use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::plugin::PluginError;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriggerEvent {
    pub resource_type: String,
    pub alias: String,
    pub event: String,
    pub payload_schema: crate::LashSchema,
}

impl TriggerEvent {
    pub fn new(
        resource_type: impl Into<String>,
        alias: impl Into<String>,
        event: impl Into<String>,
        payload_schema: crate::LashSchema,
    ) -> Self {
        Self {
            resource_type: resource_type.into(),
            alias: alias.into(),
            event: event.into(),
            payload_schema,
        }
    }

    pub fn payload_schema(&self) -> &crate::LashSchema {
        &self.payload_schema
    }

    pub fn key(&self) -> TriggerEventKey {
        TriggerEventKey {
            resource_type: self.resource_type.clone(),
            alias: self.alias.clone(),
            event: self.event.clone(),
        }
    }

    pub fn source_type(&self) -> String {
        trigger_event_type(&self.alias, &self.event)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TriggerEventKey {
    pub resource_type: String,
    pub alias: String,
    pub event: String,
}

impl TriggerEventKey {
    pub fn new(
        resource_type: impl Into<String>,
        alias: impl Into<String>,
        event: impl Into<String>,
    ) -> Self {
        Self {
            resource_type: resource_type.into(),
            alias: alias.into(),
            event: event.into(),
        }
    }

    pub fn source_type(&self) -> String {
        trigger_event_type(&self.alias, &self.event)
    }
}

pub fn trigger_event_type(alias: &str, event: &str) -> String {
    format!("{alias}.{event}")
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TriggerEventCatalog {
    events: BTreeMap<TriggerEventKey, TriggerEvent>,
}

impl TriggerEventCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn declare(&mut self, event: TriggerEvent) -> Result<(), String> {
        let key = event.key();
        if self.events.contains_key(&key) {
            return Err(format!(
                "duplicate trigger occurrence `{}.{}.{}`",
                key.resource_type, key.alias, key.event
            ));
        }
        let source_type = event.source_type();
        if let Some(existing) = self
            .events
            .values()
            .find(|existing| existing.source_type() == source_type)
        {
            return Err(format!(
                "duplicate trigger source `{source_type}` declared by `{}.{}.{}` and `{}.{}.{}`",
                existing.resource_type,
                existing.alias,
                existing.event,
                key.resource_type,
                key.alias,
                key.event
            ));
        }
        self.events.insert(key, event);
        Ok(())
    }

    pub fn from_events(events: impl IntoIterator<Item = TriggerEvent>) -> Result<Self, String> {
        let mut catalog = Self::new();
        for event in events {
            catalog.declare(event)?;
        }
        Ok(catalog)
    }

    pub fn get(&self, resource_type: &str, alias: &str, event: &str) -> Option<&TriggerEvent> {
        self.events
            .get(&TriggerEventKey::new(resource_type, alias, event))
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn events(&self) -> impl Iterator<Item = &TriggerEvent> {
        self.events.values()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriggerEmitReport {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub occurrence_id: String,
    pub started_process_ids: Vec<String>,
}

impl TriggerEmitReport {
    pub fn empty() -> Self {
        Self::default()
    }

    fn new(occurrence_id: String, started_process_ids: Vec<String>) -> Self {
        Self {
            occurrence_id,
            started_process_ids,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriggerOccurrenceRequest {
    pub source_type: String,
    pub source_key: String,
    #[serde(default)]
    pub payload: serde_json::Value,
    pub idempotency_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<serde_json::Value>,
}

impl TriggerOccurrenceRequest {
    pub fn new(
        source_type: impl Into<String>,
        source_key: impl Into<String>,
        payload: serde_json::Value,
        idempotency_key: impl Into<String>,
    ) -> Self {
        Self {
            source_type: source_type.into(),
            source_key: source_key.into(),
            payload,
            idempotency_key: idempotency_key.into(),
            source: None,
        }
    }

    pub fn with_source(mut self, source: serde_json::Value) -> Self {
        self.source = Some(source);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriggerOccurrenceRecord {
    pub occurrence_id: String,
    pub source_type: String,
    pub source_key: String,
    #[serde(default)]
    pub payload: serde_json::Value,
    pub idempotency_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<serde_json::Value>,
    pub occurred_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TriggerEventType(String);

impl TriggerEventType {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for TriggerEventType {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for TriggerEventType {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl AsRef<str> for TriggerEventType {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for TriggerEventType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriggerRegistration {
    pub handle: String,
    pub source_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub source_type: TriggerEventType,
    pub source: serde_json::Value,
    pub target: TriggerTargetSummary,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriggerTargetSummary {
    pub label: Option<String>,
    pub identity: crate::ProcessIdentity,
    pub input: crate::ProcessInput,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub inputs: BTreeMap<String, TriggerInputBinding>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TriggerInputBinding {
    Event,
    Fixed { value: serde_json::Value },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriggerSubscriptionDraft {
    pub registrant: crate::ProcessOriginator,
    pub env_ref: crate::ProcessExecutionEnvRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake_target: Option<crate::SessionScope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub source_type: String,
    pub source_key: String,
    pub source: serde_json::Value,
    pub payload_schema: crate::LashSchema,
    pub target: crate::ProcessInput,
    pub target_identity: crate::ProcessIdentity,
    #[serde(default)]
    pub event_types: Vec<crate::ProcessEventType>,
    #[serde(default)]
    pub input_template: BTreeMap<String, TriggerInputBinding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_label: Option<String>,
}

impl TriggerSubscriptionDraft {
    pub fn for_process(
        registrant: crate::ProcessOriginator,
        env_ref: crate::ProcessExecutionEnvRef,
        source_type: impl Into<String>,
        source_key: impl Into<String>,
        target: crate::ProcessInput,
        target_identity: crate::ProcessIdentity,
    ) -> Self {
        let target_label = target_identity.label.clone();
        Self {
            registrant,
            env_ref,
            wake_target: None,
            name: None,
            source_type: source_type.into(),
            source_key: source_key.into(),
            source: serde_json::Value::Object(serde_json::Map::new()),
            payload_schema: crate::LashSchema::new(serde_json::Value::Object(
                serde_json::Map::new(),
            )),
            target,
            target_identity,
            event_types: Vec::new(),
            input_template: BTreeMap::new(),
            target_label,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_source(mut self, source: serde_json::Value) -> Self {
        self.source = source;
        self
    }

    pub fn with_payload_schema(mut self, payload_schema: crate::LashSchema) -> Self {
        self.payload_schema = payload_schema;
        self
    }

    pub fn with_wake_target(mut self, wake_target: crate::SessionScope) -> Self {
        self.wake_target = Some(wake_target);
        self
    }

    pub fn with_event_types(
        mut self,
        event_types: impl IntoIterator<Item = crate::ProcessEventType>,
    ) -> Self {
        self.event_types = event_types.into_iter().collect();
        self
    }

    pub fn with_input_template(
        mut self,
        input_template: BTreeMap<String, TriggerInputBinding>,
    ) -> Self {
        self.input_template = input_template;
        self
    }

    pub fn with_target_label(mut self, target_label: impl Into<String>) -> Self {
        self.target_label = Some(target_label.into());
        self
    }

    pub fn validate(&self) -> Result<(), PluginError> {
        validate_trigger_subscription_target_label(
            self.target_label.as_deref(),
            self.target_identity.label.as_deref(),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriggerSubscriptionRecord {
    pub subscription_id: String,
    pub registrant: crate::ProcessOriginator,
    pub env_ref: crate::ProcessExecutionEnvRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake_target: Option<crate::SessionScope>,
    pub handle: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub source_type: String,
    pub source_key: String,
    pub source: serde_json::Value,
    pub payload_schema: crate::LashSchema,
    pub target: crate::ProcessInput,
    pub target_identity: crate::ProcessIdentity,
    #[serde(default)]
    pub event_types: Vec<crate::ProcessEventType>,
    #[serde(default)]
    pub input_template: BTreeMap<String, TriggerInputBinding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_label: Option<String>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

impl TriggerSubscriptionRecord {
    pub fn registrant_scope_id(&self) -> String {
        self.registrant.scope_id()
    }

    pub fn registrant_session_id(&self) -> Option<&str> {
        match &self.registrant {
            crate::ProcessOriginator::Session { scope } => Some(scope.session_id.as_str()),
            crate::ProcessOriginator::Host => None,
        }
    }
}

fn validate_trigger_subscription_target_label(
    target_label: Option<&str>,
    identity_label: Option<&str>,
) -> Result<(), PluginError> {
    match (target_label, identity_label) {
        (Some(target_label), Some(identity_label)) if target_label != identity_label => {
            Err(PluginError::Session(
                "trigger target_label must match target_identity.label when both are present"
                    .to_string(),
            ))
        }
        _ => Ok(()),
    }
}

impl From<&TriggerSubscriptionRecord> for TriggerRegistration {
    fn from(route: &TriggerSubscriptionRecord) -> Self {
        Self {
            handle: route.handle.clone(),
            source_key: route.source_key.clone(),
            name: route.name.clone(),
            source_type: TriggerEventType::new(route.source_type.clone()),
            source: route.source.clone(),
            target: TriggerTargetSummary {
                label: route.target_label.clone(),
                identity: route.target_identity.clone(),
                input: route.target.clone(),
                inputs: route.input_template.clone(),
            },
            enabled: route.enabled,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TriggerSubscriptionFilter {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handle: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

impl TriggerSubscriptionFilter {
    pub fn for_session(session_id: impl Into<String>) -> Self {
        Self {
            session_id: Some(session_id.into()),
            ..Self::default()
        }
    }

    pub fn for_source_type(source_type: impl Into<String>) -> Self {
        Self {
            source_type: Some(source_type.into()),
            ..Self::default()
        }
    }

    pub fn matches(&self, record: &TriggerSubscriptionRecord) -> bool {
        self.session_id
            .as_deref()
            .is_none_or(|session_id| record.registrant_session_id() == Some(session_id))
            && self
                .handle
                .as_deref()
                .is_none_or(|handle| record.handle == handle)
            && self
                .name
                .as_deref()
                .is_none_or(|name| record.name.as_deref() == Some(name))
            && self
                .source_type
                .as_deref()
                .is_none_or(|source_type| record.source_type == source_type)
            && self
                .source_key
                .as_deref()
                .is_none_or(|source_key| record.source_key == source_key)
            && self.enabled.is_none_or(|enabled| record.enabled == enabled)
            && self
                .target
                .as_ref()
                .is_none_or(|target| record.target_identity.definition.as_ref() == Some(target))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TriggerDeliveryReservation {
    pub occurrence: TriggerOccurrenceRecord,
    pub subscription: TriggerSubscriptionRecord,
    pub process_id: String,
}

#[async_trait::async_trait]
pub trait TriggerStore: Send + Sync {
    fn durability_tier(&self) -> crate::DurabilityTier {
        crate::DurabilityTier::Inline
    }

    async fn source_key_for_subscription(
        &self,
        source_type: &str,
        source: &serde_json::Value,
    ) -> Result<String, PluginError> {
        default_trigger_source_key(source_type, source)
    }

    async fn register_subscription(
        &self,
        draft: TriggerSubscriptionDraft,
    ) -> Result<TriggerSubscriptionRecord, PluginError>;

    async fn list_subscriptions(
        &self,
        filter: TriggerSubscriptionFilter,
    ) -> Result<Vec<TriggerSubscriptionRecord>, PluginError>;

    async fn cancel_subscription(
        &self,
        session_id: &str,
        handle: &str,
    ) -> Result<bool, PluginError>;

    async fn delete_session_subscriptions(&self, session_id: &str) -> Result<usize, PluginError>;

    async fn record_occurrence(
        &self,
        request: TriggerOccurrenceRequest,
    ) -> Result<TriggerOccurrenceRecord, PluginError>;

    async fn reserve_matching_deliveries(
        &self,
        occurrence_id: &str,
    ) -> Result<Vec<TriggerDeliveryReservation>, PluginError>;
}

pub struct InMemoryTriggerStore {
    clock: Arc<dyn crate::Clock>,
    state: Mutex<InMemoryTriggerEventState>,
}

impl InMemoryTriggerStore {
    pub fn new() -> Self {
        Self::with_clock(Arc::new(crate::SystemClock))
    }

    pub fn with_clock(clock: Arc<dyn crate::Clock>) -> Self {
        Self {
            clock,
            state: Mutex::new(InMemoryTriggerEventState::default()),
        }
    }
}

impl Default for InMemoryTriggerStore {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default)]
struct InMemoryTriggerEventState {
    next_subscription_seq: u64,
    subscriptions: BTreeMap<String, TriggerSubscriptionRecord>,
    occurrences: BTreeMap<String, TriggerOccurrenceRecord>,
    occurrence_id_by_idempotency_key: BTreeMap<String, String>,
    occurrence_hashes: BTreeMap<String, String>,
    deliveries: BTreeSet<(String, String)>,
}

#[async_trait::async_trait]
impl TriggerStore for InMemoryTriggerStore {
    async fn register_subscription(
        &self,
        draft: TriggerSubscriptionDraft,
    ) -> Result<TriggerSubscriptionRecord, PluginError> {
        draft.validate()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger store lock poisoned".to_string()))?;
        state.next_subscription_seq = state.next_subscription_seq.saturating_add(1);
        let handle = format!("trigger:{}", state.next_subscription_seq);
        let subscription_id = format!("subscription:{}", state.next_subscription_seq);
        let now = self.clock.timestamp_ms();
        let record = TriggerSubscriptionRecord {
            subscription_id: subscription_id.clone(),
            registrant: draft.registrant,
            env_ref: draft.env_ref,
            wake_target: draft.wake_target,
            handle,
            name: draft.name,
            source_type: draft.source_type,
            source_key: draft.source_key,
            source: draft.source,
            payload_schema: draft.payload_schema,
            target: draft.target,
            target_identity: draft.target_identity,
            event_types: draft.event_types,
            input_template: draft.input_template,
            target_label: draft.target_label,
            enabled: true,
            created_at_ms: now,
            updated_at_ms: now,
        };
        state.subscriptions.insert(subscription_id, record.clone());
        Ok(record)
    }

    async fn list_subscriptions(
        &self,
        filter: TriggerSubscriptionFilter,
    ) -> Result<Vec<TriggerSubscriptionRecord>, PluginError> {
        let state = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger store lock poisoned".to_string()))?;
        let mut records = state
            .subscriptions
            .values()
            .filter(|record| filter.matches(record))
            .cloned()
            .collect::<Vec<_>>();
        records.sort_by(|left, right| {
            left.registrant_scope_id()
                .cmp(&right.registrant_scope_id())
                .then_with(|| left.handle.cmp(&right.handle))
        });
        Ok(records)
    }

    async fn cancel_subscription(
        &self,
        session_id: &str,
        handle: &str,
    ) -> Result<bool, PluginError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger store lock poisoned".to_string()))?;
        let now = self.clock.timestamp_ms();
        let Some(record) = state.subscriptions.values_mut().find(|record| {
            record.registrant_session_id() == Some(session_id) && record.handle == handle
        }) else {
            return Ok(false);
        };
        let changed = record.enabled;
        record.enabled = false;
        record.updated_at_ms = now;
        Ok(changed)
    }

    async fn delete_session_subscriptions(&self, session_id: &str) -> Result<usize, PluginError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger store lock poisoned".to_string()))?;
        let before = state.subscriptions.len();
        state
            .subscriptions
            .retain(|_, record| record.registrant_session_id() != Some(session_id));
        Ok(before.saturating_sub(state.subscriptions.len()))
    }

    async fn record_occurrence(
        &self,
        request: TriggerOccurrenceRequest,
    ) -> Result<TriggerOccurrenceRecord, PluginError> {
        validate_trigger_occurrence_request(&request)?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger store lock poisoned".to_string()))?;
        let request_hash = trigger_occurrence_request_hash(&request)?;
        if let Some(existing_id) = state
            .occurrence_id_by_idempotency_key
            .get(&request.idempotency_key)
            .cloned()
        {
            let existing_hash = state
                .occurrence_hashes
                .get(&existing_id)
                .cloned()
                .unwrap_or_default();
            if existing_hash != request_hash {
                return Err(PluginError::Session(format!(
                    "trigger occurrence idempotency conflict for `{}`",
                    request.idempotency_key
                )));
            }
            return state.occurrences.get(&existing_id).cloned().ok_or_else(|| {
                PluginError::Session(format!(
                    "missing trigger occurrence `{existing_id}` for idempotency key"
                ))
            });
        }
        let occurrence_id = deterministic_occurrence_id(&request)?;
        let record = TriggerOccurrenceRecord {
            occurrence_id: occurrence_id.clone(),
            source_type: request.source_type,
            source_key: request.source_key,
            payload: request.payload,
            idempotency_key: request.idempotency_key.clone(),
            source: request.source,
            occurred_at_ms: self.clock.timestamp_ms(),
        };
        state
            .occurrence_id_by_idempotency_key
            .insert(request.idempotency_key, occurrence_id.clone());
        state
            .occurrence_hashes
            .insert(occurrence_id.clone(), request_hash);
        state.occurrences.insert(occurrence_id, record.clone());
        Ok(record)
    }

    async fn reserve_matching_deliveries(
        &self,
        occurrence_id: &str,
    ) -> Result<Vec<TriggerDeliveryReservation>, PluginError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger store lock poisoned".to_string()))?;
        let occurrence = state
            .occurrences
            .get(occurrence_id)
            .cloned()
            .ok_or_else(|| {
                PluginError::Session(format!("unknown trigger occurrence `{occurrence_id}`"))
            })?;
        let subscriptions = state
            .subscriptions
            .values()
            .filter(|record| {
                record.enabled
                    && record.source_type == occurrence.source_type
                    && record.source_key == occurrence.source_key
            })
            .cloned()
            .collect::<Vec<_>>();
        let mut deliveries = Vec::new();
        for subscription in subscriptions {
            let key = (
                occurrence.occurrence_id.clone(),
                subscription.subscription_id.clone(),
            );
            if !state.deliveries.insert(key) {
                continue;
            }
            let process_id = deterministic_delivery_process_id(
                &occurrence.occurrence_id,
                &subscription.subscription_id,
            )?;
            deliveries.push(TriggerDeliveryReservation {
                occurrence: occurrence.clone(),
                subscription,
                process_id,
            });
        }
        Ok(deliveries)
    }
}

fn default_enabled() -> bool {
    true
}

pub fn default_trigger_source_key(
    source_type: &str,
    source: &serde_json::Value,
) -> Result<String, PluginError> {
    let digest = crate::stable_hash::stable_json_sha256_hex(&(source_type, source))
        .map_err(|err| PluginError::Session(format!("failed to hash trigger source key: {err}")))?;
    Ok(format!("source:{source_type}:sha256:{digest}"))
}

pub fn empty_trigger_source_key(source_type: &str) -> Result<String, PluginError> {
    default_trigger_source_key(source_type, &serde_json::json!({}))
}

pub fn deterministic_occurrence_id(
    request: &TriggerOccurrenceRequest,
) -> Result<String, PluginError> {
    let digest = crate::stable_hash::stable_json_sha256_hex(&(
        request.source_type.as_str(),
        request.source_key.as_str(),
        request.idempotency_key.as_str(),
    ))
    .map_err(|err| PluginError::Session(format!("failed to hash trigger occurrence: {err}")))?;
    Ok(format!("trigger:{digest}"))
}

pub fn deterministic_delivery_process_id(
    occurrence_id: &str,
    subscription_id: &str,
) -> Result<String, PluginError> {
    let digest = crate::stable_hash::stable_json_sha256_hex(&(occurrence_id, subscription_id))
        .map_err(|err| PluginError::Session(format!("failed to hash trigger delivery: {err}")))?;
    Ok(format!("process:trigger:{digest}"))
}

#[derive(Clone)]
pub struct TriggerRouter {
    store: Arc<dyn TriggerStore>,
    process_registry: Option<Arc<dyn crate::ProcessRegistry>>,
    process_work_driver: Option<crate::ProcessWorkDriver>,
}

impl TriggerRouter {
    pub fn new(
        store: Arc<dyn TriggerStore>,
        process_registry: Option<Arc<dyn crate::ProcessRegistry>>,
        process_work_driver: Option<crate::ProcessWorkDriver>,
    ) -> Self {
        Self {
            store,
            process_registry,
            process_work_driver,
        }
    }

    pub fn store(&self) -> Arc<dyn TriggerStore> {
        Arc::clone(&self.store)
    }

    pub async fn emit(
        &self,
        request: TriggerOccurrenceRequest,
        effect_controller: &dyn crate::RuntimeEffectController,
    ) -> Result<TriggerEmitReport, PluginError> {
        let occurrence = self.store.record_occurrence(request).await?;
        let reservations = self
            .store
            .reserve_matching_deliveries(&occurrence.occurrence_id)
            .await?;
        let Some(process_registry) = self.process_registry.as_ref() else {
            if reservations.is_empty() {
                return Ok(TriggerEmitReport::new(occurrence.occurrence_id, Vec::new()));
            }
            return Err(PluginError::Session(
                "trigger delivery requires a process registry".to_string(),
            ));
        };
        let mut started_process_ids = Vec::new();
        let mut start_errors = Vec::new();
        for reservation in reservations {
            let process_id = reservation.process_id.clone();
            if let Err(err) = self
                .start_delivery(
                    &reservation,
                    Arc::clone(process_registry),
                    effect_controller,
                )
                .await
            {
                start_errors.push(format!(
                    "{}: {err}",
                    reservation.subscription.subscription_id
                ));
                continue;
            }
            started_process_ids.push(process_id);
        }
        if !started_process_ids.is_empty()
            && let Some(driver) = self.process_work_driver.as_ref()
        {
            driver.claim_and_run_pending("trigger_delivery").await?;
        }
        if started_process_ids.is_empty()
            && let Some(message) = trigger_delivery_failure_summary(&start_errors)
        {
            return Err(PluginError::Session(message));
        }
        Ok(TriggerEmitReport::new(
            occurrence.occurrence_id,
            started_process_ids,
        ))
    }

    async fn start_delivery(
        &self,
        reservation: &TriggerDeliveryReservation,
        process_registry: Arc<dyn crate::ProcessRegistry>,
        effect_controller: &dyn crate::RuntimeEffectController,
    ) -> Result<(), PluginError> {
        let subscription = &reservation.subscription;
        let occurrence = &reservation.occurrence;
        subscription
            .payload_schema
            .validate(&occurrence.payload)
            .map_err(|err| {
                PluginError::Session(format!(
                    "invalid payload for trigger `{}`: {err}",
                    subscription.handle
                ))
            })?;
        let args =
            materialize_trigger_process_args(&subscription.input_template, &occurrence.payload)?;
        let target = apply_trigger_inputs(subscription.target.clone(), args)?;
        let originator_scope_id = subscription.registrant_scope_id();
        let trigger_occurrence_invocation = crate::runtime::causal::trigger_occurrence_invocation(
            &originator_scope_id,
            &occurrence.occurrence_id,
        );
        let registration = crate::ProcessRegistration::new(
            reservation.process_id.clone(),
            target.clone(),
            crate::ProcessProvenance::new(subscription.registrant.clone())
                .with_caused_by(trigger_occurrence_invocation.causal_ref()),
        )
        .with_identity(subscription.target_identity.clone())
        .with_extra_event_types(subscription.event_types.clone())
        .with_execution_env_ref(Some(subscription.env_ref.clone()))
        .with_wake_target(subscription.wake_target.clone());
        let descriptor_kind = subscription.target_identity.kind.clone();
        let grant =
            subscription
                .wake_target
                .clone()
                .map(|session_scope| crate::ProcessStartGrant {
                    session_scope,
                    descriptor: crate::ProcessHandleDescriptor::new(
                        Some(descriptor_kind.as_str()),
                        subscription.target_label.as_deref(),
                    ),
                });
        let execution_context = crate::ProcessExecutionContext::default()
            .with_causal_invocation(Some(trigger_occurrence_invocation));
        let command = crate::ProcessCommand::Start {
            registration,
            grant,
            execution_context: Box::new(execution_context),
        };
        let effect_id = command.effect_id();
        let invocation = crate::RuntimeInvocation::effect(
            crate::RuntimeScope::new(originator_scope_id),
            effect_id.clone(),
            crate::RuntimeEffectKind::Process,
            format!(
                "trigger:{}:{}",
                occurrence.occurrence_id, subscription.subscription_id
            ),
        )
        .with_caused_by(Some(crate::CausalRef::TriggerOccurrence {
            occurrence_id: occurrence.occurrence_id.clone(),
        }));
        let outcome = effect_controller
            .execute_effect(
                crate::RuntimeEffectEnvelope::new(
                    invocation,
                    crate::RuntimeEffectCommand::process(command),
                ),
                crate::RuntimeEffectLocalExecutor::processes(
                    process_registry,
                    self.process_work_driver.clone(),
                ),
            )
            .await?;
        match outcome {
            crate::RuntimeEffectOutcome::Process {
                result: crate::ProcessEffectOutcome::Start { .. },
            } => Ok(()),
            other => Err(PluginError::Session(format!(
                "trigger process start returned the wrong outcome: {}",
                other.kind().as_str()
            ))),
        }
    }
}

fn trigger_delivery_failure_summary(errors: &[String]) -> Option<String> {
    match errors {
        [] => None,
        [only] => Some(format!("trigger delivery failed: {only}")),
        [first, rest @ ..] => Some(format!(
            "trigger delivery failed for {} matching subscriptions: {first}; {} more failed",
            errors.len(),
            rest.len()
        )),
    }
}

fn materialize_trigger_process_args(
    input_template: &BTreeMap<String, TriggerInputBinding>,
    event_payload: &serde_json::Value,
) -> Result<serde_json::Map<String, serde_json::Value>, PluginError> {
    let mut args = serde_json::Map::new();
    for (input_name, input) in input_template {
        let value = match input {
            TriggerInputBinding::Event => event_payload.clone(),
            TriggerInputBinding::Fixed { value } => value.clone(),
        };
        args.insert(input_name.to_string(), value);
    }
    Ok(args)
}

fn apply_trigger_inputs(
    mut target: crate::ProcessInput,
    args: serde_json::Map<String, serde_json::Value>,
) -> Result<crate::ProcessInput, PluginError> {
    match &mut target {
        crate::ProcessInput::Engine { payload, .. } => {
            let object = payload.as_object_mut().ok_or_else(|| {
                PluginError::Session(
                    "trigger engine target payload must be a JSON object".to_string(),
                )
            })?;
            object.insert("args".to_string(), serde_json::Value::Object(args));
            Ok(target)
        }
        other => Err(PluginError::Session(format!(
            "trigger target must be an engine process, got {}",
            other.engine_kind()
        ))),
    }
}

pub fn validate_trigger_occurrence_request(
    request: &TriggerOccurrenceRequest,
) -> Result<(), PluginError> {
    if request.source_type.trim().is_empty() {
        return Err(PluginError::Session(
            "trigger occurrence requires source_type".to_string(),
        ));
    }
    if request.source_key.trim().is_empty() {
        return Err(PluginError::Session(
            "trigger occurrence requires source_key".to_string(),
        ));
    }
    if request.idempotency_key.trim().is_empty() {
        return Err(PluginError::Session(
            "trigger occurrence requires idempotency_key".to_string(),
        ));
    }
    Ok(())
}

pub fn trigger_occurrence_request_hash(
    request: &TriggerOccurrenceRequest,
) -> Result<String, PluginError> {
    crate::stable_hash::stable_json_sha256_hex(&(
        request.source_type.as_str(),
        request.source_key.as_str(),
        &request.payload,
        &request.source,
    ))
    .map_err(|err| PluginError::Session(format!("failed to hash trigger occurrence: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn button_payload_schema() -> crate::LashSchema {
        crate::LashSchema::any()
    }

    #[test]
    fn trigger_catalog_rejects_duplicate_trigger_source_identity() {
        let mut catalog = TriggerEventCatalog::new();
        catalog
            .declare(TriggerEvent::new(
                "Button",
                "ui.button",
                "pressed",
                button_payload_schema(),
            ))
            .expect("first trigger occurrence");

        let err = catalog
            .declare(TriggerEvent::new(
                "AlternateButton",
                "ui.button",
                "pressed",
                button_payload_schema(),
            ))
            .expect_err("duplicate public source identity should be rejected");

        assert!(err.contains("duplicate trigger source `ui.button.pressed`"));
    }

    #[tokio::test]
    async fn trigger_store_rejects_mismatched_target_label() {
        let store = InMemoryTriggerStore::default();
        let draft = TriggerSubscriptionDraft::for_process(
            crate::ProcessOriginator::host(),
            crate::ProcessExecutionEnvRef::new("process-env:test"),
            "ui.button.pressed",
            "source-key",
            crate::ProcessInput::External {
                metadata: serde_json::json!({}),
            },
            crate::ProcessIdentity::new("external").with_label(Some("expected")),
        )
        .with_target_label("other");

        let err = store
            .register_subscription(draft)
            .await
            .expect_err("mismatched target labels should be rejected");
        assert!(err.to_string().contains("target_label must match"));
    }
}
