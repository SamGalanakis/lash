use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostEvent {
    pub resource_type: String,
    pub alias: String,
    pub event: String,
    pub payload_ty: lashlang::NamedDataType,
}

impl HostEvent {
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

    pub fn key(&self) -> HostEventKey {
        HostEventKey {
            resource_type: self.resource_type.clone(),
            alias: self.alias.clone(),
            event: self.event.clone(),
        }
    }

    pub fn source_type(&self) -> String {
        host_event_source_type(&self.alias, &self.event)
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

    pub fn source_type(&self) -> String {
        host_event_source_type(&self.alias, &self.event)
    }
}

pub fn host_event_source_type(alias: &str, event: &str) -> String {
    format!("{alias}.{event}")
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
        let source_type = event.source_type();
        if let Some(existing) = self
            .events
            .values()
            .find(|existing| existing.source_type() == source_type)
        {
            return Err(format!(
                "duplicate host event source `{source_type}` declared by `{}.{}.{}` and `{}.{}.{}`",
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
pub struct HostEventEmitReport {
    pub started_process_ids: Vec<String>,
}

impl HostEventEmitReport {
    pub fn empty() -> Self {
        Self::default()
    }
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
        .expect("valid host event payload")
    }

    #[test]
    fn host_event_catalog_rejects_duplicate_trigger_source_identity() {
        let mut catalog = HostEventCatalog::new();
        catalog
            .declare(HostEvent::new(
                "Button",
                "ui.button",
                "pressed",
                button_payload_type(),
            ))
            .expect("first host event");

        let err = catalog
            .declare(HostEvent::new(
                "AlternateButton",
                "ui.button",
                "pressed",
                button_payload_type(),
            ))
            .expect_err("duplicate public source identity should be rejected");

        assert!(err.contains("duplicate host event source `ui.button.pressed`"));
    }
}
