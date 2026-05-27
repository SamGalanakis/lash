use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostEvent {
    pub resource_type: String,
    pub alias: String,
    pub event: String,
    pub payload_ty: lashlang::TypeExpr,
}

impl HostEvent {
    pub fn new(
        resource_type: impl Into<String>,
        alias: impl Into<String>,
        event: impl Into<String>,
    ) -> Self {
        Self {
            resource_type: resource_type.into(),
            alias: alias.into(),
            event: event.into(),
            payload_ty: lashlang::TypeExpr::Any,
        }
    }

    pub fn payload(mut self, payload_ty: lashlang::TypeExpr) -> Self {
        self.payload_ty = payload_ty;
        self
    }

    pub fn key(&self) -> HostEventKey {
        HostEventKey {
            resource_type: self.resource_type.clone(),
            alias: self.alias.clone(),
            event: self.event.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HostEventKey {
    pub resource_type: String,
    pub alias: String,
    pub event: String,
}

impl HostEventKey {
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
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostEventCatalog {
    events: BTreeMap<HostEventKey, HostEvent>,
}

impl HostEventCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn declare(&mut self, event: HostEvent) -> Result<(), String> {
        let key = event.key();
        if self.events.contains_key(&key) {
            return Err(format!(
                "duplicate host event `{}.{}.{}`",
                key.resource_type, key.alias, key.event
            ));
        }
        self.events.insert(key, event);
        Ok(())
    }

    pub fn from_events(events: impl IntoIterator<Item = HostEvent>) -> Result<Self, String> {
        let mut catalog = Self::new();
        for event in events {
            catalog.declare(event)?;
        }
        Ok(catalog)
    }

    pub fn get(&self, resource_type: &str, alias: &str, event: &str) -> Option<&HostEvent> {
        self.events
            .get(&HostEventKey::new(resource_type, alias, event))
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn events(&self) -> impl Iterator<Item = &HostEvent> {
        self.events.values()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionTriggerInstallReport {
    pub installed: Vec<String>,
    pub replaced: Vec<String>,
    pub unchanged: Vec<String>,
}

impl SessionTriggerInstallReport {
    pub fn trigger_names(&self) -> Vec<String> {
        self.installed
            .iter()
            .chain(self.replaced.iter())
            .chain(self.unchanged.iter())
            .cloned()
            .collect()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostEventEmitReport {
    pub started_process_ids: Vec<String>,
}

impl HostEventEmitReport {
    pub fn empty() -> Self {
        Self::default()
    }
}
