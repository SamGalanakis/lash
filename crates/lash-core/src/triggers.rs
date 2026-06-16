use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::plugin::PluginError;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriggerEvent {
    pub resource_type: String,
    pub alias: String,
    pub event: String,
    pub payload_ty: lashlang::NamedDataType,
}

impl TriggerEvent {
    pub fn new(
        resource_type: impl Into<String>,
        alias: impl Into<String>,
        event: impl Into<String>,
        payload_ty: lashlang::NamedDataType,
    ) -> Self {
        Self {
            resource_type: resource_type.into(),
            alias: alias.into(),
            event: event.into(),
            payload_ty,
        }
    }

    pub fn payload_type(&self) -> &lashlang::NamedDataType {
        &self.payload_ty
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

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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
    pub process_name: String,
    pub inputs: lashlang::TriggerInputTemplate,
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
    pub event_ty: lashlang::TypeExpr,
    pub module_ref: lashlang::ModuleRef,
    pub host_requirements_ref: lashlang::HostRequirementsRef,
    pub process_ref: lashlang::ProcessRef,
    pub process_name: String,
    pub input_template: lashlang::TriggerInputTemplate,
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
    pub event_ty: lashlang::TypeExpr,
    pub module_ref: lashlang::ModuleRef,
    pub host_requirements_ref: lashlang::HostRequirementsRef,
    pub process_ref: lashlang::ProcessRef,
    pub process_name: String,
    pub input_template: lashlang::TriggerInputTemplate,
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

impl From<&TriggerSubscriptionRecord> for TriggerRegistration {
    fn from(route: &TriggerSubscriptionRecord) -> Self {
        Self {
            handle: route.handle.clone(),
            source_key: route.source_key.clone(),
            name: route.name.clone(),
            source_type: TriggerEventType::new(route.source_type.clone()),
            source: route.source.clone(),
            target: TriggerTargetSummary {
                process_name: route.process_name.clone(),
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
    pub target: Option<lashlang::ProcessDefinitionIdentity>,
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
            && self.target.as_ref().is_none_or(|target| {
                target.matches_input_refs(
                    &record.module_ref,
                    &record.host_requirements_ref,
                    &record.process_ref,
                    &record.process_name,
                )
            })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

#[derive(Default)]
pub struct InMemoryTriggerStore {
    state: Mutex<InMemoryTriggerEventState>,
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
        let mut state = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger store lock poisoned".to_string()))?;
        state.next_subscription_seq = state.next_subscription_seq.saturating_add(1);
        let handle = format!("trigger:{}", state.next_subscription_seq);
        let subscription_id = format!("subscription:{}", state.next_subscription_seq);
        let now = crate::runtime::current_epoch_ms();
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
            event_ty: draft.event_ty,
            module_ref: draft.module_ref,
            host_requirements_ref: draft.host_requirements_ref,
            process_ref: draft.process_ref,
            process_name: draft.process_name,
            input_template: draft.input_template,
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
        let now = crate::runtime::current_epoch_ms();
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
            occurred_at_ms: crate::runtime::current_epoch_ms(),
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
    artifact_store: Arc<dyn lashlang::LashlangArtifactStore>,
    process_registry: Option<Arc<dyn crate::ProcessRegistry>>,
    process_work_poke: Option<crate::ProcessWorkPoke>,
}

impl TriggerRouter {
    pub fn new(
        store: Arc<dyn TriggerStore>,
        artifact_store: Arc<dyn lashlang::LashlangArtifactStore>,
        process_registry: Option<Arc<dyn crate::ProcessRegistry>>,
        process_work_poke: Option<crate::ProcessWorkPoke>,
    ) -> Self {
        Self {
            store,
            artifact_store,
            process_registry,
            process_work_poke,
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
            && let Some(poke) = self.process_work_poke.as_ref()
        {
            poke.poke();
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
        validate_payload(&occurrence.payload, &subscription.event_ty).map_err(|message| {
            PluginError::Session(format!(
                "invalid payload for trigger `{}`: {message}",
                subscription.handle
            ))
        })?;
        let artifact = self
            .artifact_store
            .get_module_artifact(&subscription.module_ref)
            .await
            .map_err(|err| {
                PluginError::Session(format!(
                    "failed to load trigger target module `{}`: {err}",
                    subscription.module_ref
                ))
            })?
            .ok_or_else(|| {
                PluginError::Session(format!(
                    "missing trigger target module `{}`",
                    subscription.module_ref
                ))
            })?;
        let signal_event_types = artifact
            .canonical_ir
            .process(&subscription.process_name)
            .map(crate::lashlang_process_signal_event_types)
            .unwrap_or_default();
        let args =
            materialize_trigger_process_args(&subscription.input_template, &occurrence.payload)?;
        let originator_scope_id = subscription.registrant_scope_id();
        let trigger_occurrence_invocation = crate::runtime::causal::trigger_occurrence_invocation(
            &originator_scope_id,
            &occurrence.occurrence_id,
        );
        let registration = crate::ProcessRegistration::new(
            reservation.process_id.clone(),
            crate::ProcessInput::LashlangProcess {
                module_ref: subscription.module_ref.clone(),
                process_ref: subscription.process_ref.clone(),
                host_requirements_ref: subscription.host_requirements_ref.clone(),
                process_name: subscription.process_name.clone(),
                args,
            },
            crate::ProcessProvenance::new(subscription.registrant.clone())
                .with_caused_by(trigger_occurrence_invocation.causal_ref()),
        )
        .with_extra_event_types(
            crate::lashlang_process_event_types()
                .into_iter()
                .chain(signal_event_types),
        )
        .with_execution_env_ref(Some(subscription.env_ref.clone()))
        .with_wake_target(subscription.wake_target.clone());
        let grant =
            subscription
                .wake_target
                .clone()
                .map(|session_scope| crate::ProcessStartGrant {
                    session_scope,
                    descriptor: crate::ProcessHandleDescriptor::new(
                        Some("lashlang"),
                        Some(subscription.process_name.as_str()),
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
                crate::RuntimeEffectLocalExecutor::processes(process_registry),
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
    input_template: &lashlang::TriggerInputTemplate,
    event_payload: &serde_json::Value,
) -> Result<serde_json::Map<String, serde_json::Value>, PluginError> {
    let mut args = lashlang::Record::default();
    for (input_name, input) in input_template.entries() {
        let value = match input {
            lashlang::TriggerInputBinding::Event => event_payload.clone(),
            lashlang::TriggerInputBinding::Fixed { value } => value.clone(),
        };
        args.insert(input_name.to_string(), lashlang::from_json(value));
    }
    match serde_json::to_value(lashlang::Value::Record(Arc::new(args)))
        .map_err(|err| PluginError::Session(format!("serialize trigger process args: {err}")))?
    {
        serde_json::Value::Object(map) => Ok(map),
        _ => Err(PluginError::Session(
            "trigger process args must serialize as an object".to_string(),
        )),
    }
}

pub fn validate_payload(value: &serde_json::Value, ty: &lashlang::TypeExpr) -> Result<(), String> {
    if json_matches_type(value, ty) {
        Ok(())
    } else {
        Err(format!("expected {}", lashlang::format_type_expr(ty)))
    }
}

fn json_matches_type(value: &serde_json::Value, ty: &lashlang::TypeExpr) -> bool {
    match ty {
        lashlang::TypeExpr::Any => true,
        lashlang::TypeExpr::Ref(_) => false,
        lashlang::TypeExpr::Str => value.is_string(),
        lashlang::TypeExpr::Int => value.as_i64().is_some() || value.as_u64().is_some(),
        lashlang::TypeExpr::Float => value.is_number(),
        lashlang::TypeExpr::Bool => value.is_boolean(),
        lashlang::TypeExpr::Dict => value.is_object(),
        lashlang::TypeExpr::Null => value.is_null(),
        lashlang::TypeExpr::Enum(values) => value
            .as_str()
            .is_some_and(|value| values.iter().any(|candidate| candidate.as_str() == value)),
        lashlang::TypeExpr::List(item) => value.as_array().is_some_and(|items| {
            items
                .iter()
                .all(|item_value| json_matches_type(item_value, item))
        }),
        lashlang::TypeExpr::Object(fields) => {
            let Some(map) = value.as_object() else {
                return false;
            };
            fields
                .iter()
                .all(|field| match map.get(field.name.as_str()) {
                    Some(field_value) => json_matches_type(field_value, &field.ty),
                    None => field.optional,
                })
        }
        lashlang::TypeExpr::Union(items) => items.iter().any(|item| json_matches_type(value, item)),
        lashlang::TypeExpr::Process { .. } | lashlang::TypeExpr::TriggerHandle(_) => {
            value.is_object()
        }
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

    fn button_payload_type() -> lashlang::NamedDataType {
        lashlang::NamedDataType::object(
            "ui.button.Pressed",
            vec![lashlang::TypeField {
                name: "button".into(),
                ty: lashlang::TypeExpr::Str,
                optional: false,
            }],
        )
        .expect("valid trigger occurrence payload")
    }

    #[test]
    fn trigger_catalog_rejects_duplicate_trigger_source_identity() {
        let mut catalog = TriggerEventCatalog::new();
        catalog
            .declare(TriggerEvent::new(
                "Button",
                "ui.button",
                "pressed",
                button_payload_type(),
            ))
            .expect("first trigger occurrence");

        let err = catalog
            .declare(TriggerEvent::new(
                "AlternateButton",
                "ui.button",
                "pressed",
                button_payload_type(),
            ))
            .expect_err("duplicate public source identity should be rejected");

        assert!(err.contains("duplicate trigger source `ui.button.pressed`"));
    }

    #[test]
    fn trigger_subscription_record_rejects_legacy_required_surface_ref() {
        let mut inputs = BTreeMap::new();
        inputs.insert("event".to_string(), lashlang::TriggerInputBinding::Event);
        let record = TriggerSubscriptionRecord {
            subscription_id: "subscription:1".to_string(),
            registrant: crate::ProcessOriginator::session(crate::SessionScope::new("session-a")),
            env_ref: crate::ProcessExecutionEnvRef::new("process-env:session-a"),
            wake_target: Some(crate::SessionScope::new("session-a")),
            handle: "trigger:1".to_string(),
            name: Some("button watcher".to_string()),
            source_type: "ui.button.pressed".to_string(),
            source_key: empty_trigger_source_key("ui.button.pressed").expect("source key"),
            source: serde_json::json!({}),
            event_ty: lashlang::TypeExpr::Object(vec![lashlang::TypeField {
                name: "button".into(),
                ty: lashlang::TypeExpr::Str,
                optional: false,
            }]),
            module_ref: lashlang::ModuleRef::new(&lashlang::ContentHash::new("module")),
            host_requirements_ref: lashlang::HostRequirementsRef::new(&lashlang::ContentHash::new(
                "surface",
            )),
            process_ref: lashlang::ProcessRef::new(lashlang::ContentHash::new("process"), 0),
            process_name: "on_button".to_string(),
            input_template: lashlang::TriggerInputTemplate::new(inputs),
            enabled: true,
            created_at_ms: 1,
            updated_at_ms: 1,
        };
        let mut value = serde_json::to_value(record).expect("record json");
        let object = value.as_object_mut().expect("record object");
        let legacy_ref = object
            .remove("host_requirements_ref")
            .expect("host requirements ref");
        object.insert("required_surface_ref".to_string(), legacy_ref);

        let err = serde_json::from_value::<TriggerSubscriptionRecord>(value)
            .expect_err("legacy record must be malformed");

        assert!(err.to_string().contains("host_requirements_ref"));
    }
}
